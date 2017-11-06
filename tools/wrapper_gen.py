#!/usr/bin/env python
#
# generate C++ wrappers for the LAPACK Fortran functions by parsing
# doxygen comments in LAPACK's SRC *.f files.

from __future__ import print_function

import sys
import re
import os

from text_balanced import extract_bracketed

# ------------------------------------------------------------------------------
# configuration

debug = 0
#debug = 1

typemap = {
    'character'        : 'char',
    'integer'          : 'int64_t',
    'real'             : 'float',
    'double precision' : 'double',
    'complex'          : 'std::complex<float>',
    'complex*16'       : 'std::complex<double>',
    'logical'          : 'bool',

    'logical function of two real arguments'               : 'LAPACK_S_SELECT2',
    'logical function of three real arguments'             : 'LAPACK_S_SELECT3',

    'logical function of two double precision arguments'   : 'LAPACK_D_SELECT2',
    'logical function of three double precision arguments' : 'LAPACK_D_SELECT3',

    'logical function of one complex argument'     : 'LAPACK_C_SELECT1',
    'logical function of two complex arguments'    : 'LAPACK_C_SELECT2',

    'logical function of one complex*16 argument'  : 'LAPACK_Z_SELECT1',
    'logical function of two complex*16 arguments' : 'LAPACK_Z_SELECT2',
}

enum_map = {
    'uplo'   : 'Uplo',
    'trans'  : 'Op',
    'transA' : 'Op',
    'transB' : 'Op',
    'transr' : 'Op',  # RFP
    'diag'   : 'Diag',
    'side'   : 'Side',

    'job'    : 'Job',
    'joba'   : 'Job',  # gesvj -- not compatibile with gesvd!
    'jobv'   : 'Job',  # gesvj -- not compatibile with gesvd!

    'jobu'   : 'Job',  # gesvd, gesdd; ggsvd has incompatible jobu and jobv
    'jobvt'  : 'Job',  # gesvd, gesdd
    'jobz'   : 'Job',  # syev
    'jobvl'  : 'Job',  # geev
    'jobvr'  : 'Job',  # geev
    'jobvs'  : 'Job',  # gees
    'jobvsl' : 'Job',  # gges
    'jobvsr' : 'Job',  # gges

    'jobq'   : 'JobQ',   # ggsvd

    'jobu1'  : 'JobCS',  # bbcsd
    'jobu2'  : 'JobCS',  # bbcsd
    'jobv1t' : 'JobCS',  # bbcsd
    'jobv2t' : 'JobCS',  # bbcsd

    'norm'   : 'Norm',
    'sort'   : 'Sort',        # gees
    'range'  : 'Range',       # syevx
    'vect'   : 'Vect',        # orgbr, ormbr
    'compq'  : 'CompQ',       # bdsdc, gghrd
    'compz'  : 'CompQ',       #        gghrd
    'direct' : 'Direct',      # larfb
    'storev' : 'StoreV',      # larfb
    'type'   : 'MatrixType',  # lascl
    'howmny' : 'HowMany',     # trevc
    'equed'  : 'Equed',       # *svx, *rfsx
    'fact'   : 'Factored',    # *svx
    'sense'  : 'Sense',       # geesx
    'balanc' : 'Balance',     # geevx
}

# 4 space indent
tab = '    '


# ------------------------------------------------------------------------------
# captures information about each function argument
class Arg:
    def __init__( self, name, intent ):
        self.name   = name
        self.intent = intent
        self.desc   = ''

        self.use_query = False
        self.is_array  = False
        self.dtype     = None
        self.dim       = None
        self.lbound    = None
    # end
# end

# ------------------------------------------------------------------------------
# joins a and b with ";"
def join( a, b ):
    if (a):
        return a + '; ' + b
    else:
        return b
# end

# ------------------------------------------------------------------------------
# Parses an arg's description for a dimension statement.
# Tries to find all dimensions that exist.
# Returns string of all dimensions, seperated by ";".
def parse_dim( arg ):
    if (debug): print( 'parse_dim' )
    txt = arg.desc

    # easy case: dimension at end of line
    s = re.search( r'array,\s+(dimensions?|length|size)\s+(\w+|\(.*?\))\.?\n', txt )
    if (s):
        arg.dim = s.group(2).lower()
        if (debug > 1): print( ':: easy = ' + arg.dim )
        return

    # harder cases
    # find initial dimensions keyword, and remainder after that
    state = 0
    s = re.search( r'array,\s+(dimensions?|length|size)\s*(.*)', txt, re.DOTALL )
    if (not s):
        if (debug > 1): print( ':: error: no match' )
        arg.dim = 'ERROR PARSING (1)'
        return

    arg.dim = ''
    rem = s.group(2)

    length = len(rem)
    while (length > 0):
        s = re.search( r'^(?:or|and)\s+(.*)', rem, re.DOTALL )
        if (s):
            if (debug > 1): print( ':: text = or/and' )
            rem = s.group(1)

        s = re.search( r'^at least\s*(.*)', rem, re.DOTALL )
        if (s):
            if (debug > 1): print( ':: text = at least' )
            rem = s.group(1)

        if (rem[0] == '('):
            # (paren) expression like (MAX(1,N))
            (d, rem) = extract_bracketed( rem, '(', ')' )
            if (d):
                if (debug > 1): print( ':: expr =', d )
                arg.dim = join( arg.dim, d.lower() )
                rem = rem.lstrip(' ')
            else:
                if (debug > 1): print( ':: error: bad expr' )
                arg.dim += '; ERROR PARSING (2)'
                break
        else:
            if (debug > 1): print( ':: missing expr -- implicit end' )
            break
        # end

        # multiple dimensions
        s = re.search( r'^((?:if|when|otherwise)[^\n]*)\n *(.*)', rem, re.DOTALL )
        if (s):
            if (debug > 1): print( ':: cond =', s.group(1) )
            arg.dim += ' ' + s.group(1)
            rem = s.group(2)

        # end of line
        if (re.search( r'^\.?\n', rem )):
            if (debug > 1): print( ':: eol =', arg.dim )

        # quit if no progress
        l = len(rem)
        if (l == length):
            if (debug > 1): print( ':: error: no progress' )
            arg.dim += '; ERROR PARSING (3)'
            break
        length = l
    # end
# end

# ------------------------------------------------------------------------------
# Parses an arg's description for "arg >= ..." giving lower bounds.
# Tries to find all bounds that exist.
# Returns string of all bounds, seperated by ";".
def parse_lbound( arg ):
    if (debug): print( 'parse_lbound' )

    txt = arg.desc
    lbound = ''
    found = True
    while (found):
        found = False
        s = re.search( arg.name + ' *>= *max([^\n]*)(.*)', txt, re.IGNORECASE | re.DOTALL )
        if (s):
            expr = s.group(1)
            (b, r) = extract_bracketed( expr, '(', ')' )
            if (b):
                lbound = join( lbound, 'max' + b )
                found = True
            txt = r + s.group(2)
        else:
            s = re.search( arg.name + ' *>= *([^,;.]+)(.*)', txt, re.IGNORECASE | re.DOTALL )
            if (s):
                lbound = join( lbound, s.group(1) )
                found = True
                txt = s.group(2)
    # end
    arg.lbound = lbound
# end

# ------------------------------------------------------------------------------
# state tracks what is looked for next:
# function, argument, argument's description, beginning with verbatim
state_func  = 0
state_arg   = 1
state_verb  = 2
state_desc  = 3

# ------------------------------------------------------------------------------
# Processes a single file, reading its function arguments.
# Returns a string with the wrapper.
def process( filename ):
    if (debug):
        (path, f) = os.path.split( filename )
        print( '='*60, f )

    retval = 'void'
    state  = state_func
    arg    = None
    args   = []
    is_func = False

    # --------------------
    # parse Fortran doxygen for arguments
    f = open( filename )
    for line in f:
        if (state == state_func):
            s = re.search( r'^\* +\b(.*)\b +(?:recursive +)?function +(\w+)', line, re.IGNORECASE )  #\( *(.*) *\)', line )
            if (s):
                retval  = typemap[ s.group(1).lower() ]
                func    = s.group(2).lower()
                #arglist = s.group(3)
                state   = state_arg
                is_func = True
            # end

            s = re.search( r'^\* +(?:recursive +)?subroutine +(\w+)', line, re.IGNORECASE )  #\( *(.*) *\)', line )
            if (s):
                func    = s.group(1).lower()
                #arglist = s.group(2)
                state   = state_arg
            # end
        elif (state == state_arg):
            s = re.search( r'^\*> *\\param\[(in|out|in,out|inout)\] +(\w+)', line )
            if (s):
                intent = s.group(1).lower()
                var    = s.group(2)
                arg = Arg( var, intent )
                args.append( arg )
                state = state_verb
            # end
        elif (state == state_verb):
            if (re.search( r'^\*> *\\verbatim', line )):
                state = state_desc
        elif (state == state_desc):
            if (re.search( r'^\*> *\\endverbatim', line )):
                state = state_arg
                arg = None
            else:
                arg.desc += line
        # end
    # end

    # --------------------
    # parse arguments for properties
    use_query = False
    i = 0
    for arg in args:
        if (debug): print( '-'*40 )

        arg.desc = re.sub( r'\*> *', r'', arg.desc )
        arg.desc = re.sub( r'  +', r' ', arg.desc )

        # extract data type and if array
        s = (re.search( r'^' + arg.name + ' +is +(CHARACTER|INTEGER|REAL|DOUBLE PRECISION|COMPLEX\*16|COMPLEX|LOGICAL)( array)?', arg.desc ) or
             re.search( r'^\(workspace\) +(CHARACTER|INTEGER|REAL|DOUBLE PRECISION|COMPLEX\*16|COMPLEX|LOGICAL)( array)?', arg.desc ) or
             re.search( r'^' + arg.name + ' +is a (LOGICAL FUNCTION of (?:one|two|three) (?:REAL|DOUBLE PRECISION|COMPLEX\*16|COMPLEX) arguments?)( array)?', arg.desc ))
        if (s):
            arg.dtype = typemap[ s.group(1).lower() ]
            arg.array = (s.group(2) == ' array')

        # normalize names
        # todo: in all desc?
        if (arg.name == 'CWORK'):
            arg.name = 'WORK'
            arg.desc = re.sub( '\bCWORK\b', 'WORK', arg.desc )

        if (arg.name == 'SELCTG'):
            arg.name = 'SELECT'
            arg.desc = re.sub( '\bSELCTG\b', 'SELECT', arg.desc )

        if (arg.name == 'BALANC'):
            arg.name = 'BALANCE'
            arg.desc = re.sub( '\bBALANC\b', 'BALANCE', arg.desc )

        # lowercase scalars, work and pivot arrays
        if (not arg.array or arg.name in
                ('WORK', 'SWORK', 'RWORK', 'IWORK', 'BWORK',
                'IPIV', 'JPIV', 'JPVT', 'PIV',
                'BERR', 'FERR', 'SELECT',
                'ISUPPZ', 'IFAIL', 'ISEED', 'IBLOCK', 'ISPLIT')):
            arg.name = arg.name.lower()

        # iwork, bwork needs blas_int, not int64
        if (arg.name in ('iwork', 'bwork')):
            arg.dtype = 'blas_int'

        # extract array dimensions or scalar lower bounds
        if (arg.array):
            parse_dim( arg )
            arg.dim = re.sub( 'max\( *1, *', 'max( (int64_t) 1, ', arg.dim )
        else:
            parse_lbound( arg )

        # check for workspace query
        if (arg.name in ('lwork', 'liwork', 'lrwork')):
            s = re.search( 'If ' + arg.name + ' *= *-1.*workspace query', arg.desc, re.DOTALL | re.IGNORECASE )
            if (s):
                args[i-1].use_query = True  # work
                arg.use_query = True  # lwork
                use_query = True
        # end

        # lname = local name
        # pname = pointer name to pass to Fortran
        arg.lname = arg.name
        arg.pname = arg.name
        if (arg.intent == 'in' and not arg.array and not re.search('LAPACK_._SELECT', arg.dtype)):
            # scalars, dimensions, enums
            if (arg.dtype in ('int64_t', 'bool', 'char')):
                # dimensions, enums
                arg.lname += '_'
            arg.pname = '&' + arg.lname
        elif (arg.intent in ('out', 'in,out') and not arg.array and arg.dtype in ('int64_t', 'bool')):
            # output dimensions (nfound, rank, ...)
            arg.lname += '_'
            arg.pname = '&' + arg.lname
        elif (arg.array and arg.dtype in ('int64_t', 'bool')):
            # integer or bool arrays: ipiv, select, etc.
            arg.lname += '_'
            arg.pname = arg.name + '_ptr'
        elif (re.search( 'work', arg.name )):
            arg.pname = '&' + arg.name + '[0]'
        # end

        # map char to enum (after doing lname)
        if (arg.dtype == 'char'):
            arg.dtype = 'lapack::' + enum_map[ arg.name ]

        if (debug):
            print(   'arg       = ' + arg.name +
                   '\ndtype     = ' + arg.dtype +
                   '\nintent    = ' + arg.intent +
                   '\nis_array  = ' + str(arg.is_array) +
                   '\nuse_query = ' + str(arg.use_query))
            if (arg.array):
                print( 'dim       = ' + arg.dim.lower() )
            else:
                print( 'lbound    = ' + arg.lbound.lower() )
        # end
        if (debug > 1):
            print( '\ndescription =\n' + arg.desc )
        # end

        i += 1
    # end

    # get base name of function (without precision)
    s = (re.search( '^(?:ds|zc)(gesv|posv)', func ) or
         re.search( '^[sdcz](\w+)', func ))
    if (s):
        basefunc = s.group(1)
    else:
        print( 'unknown base', func )
        basefunc = func
    # end

    # --------------------
    # build list of arguments for prototype, query, and call
    int_checks = ''
    local_vars = ''
    alloc_work = ''
    query      = ''
    cleanup    = ''
    info_throw = ''
    info_return = ''
    proto_args = []
    query_args = []
    call_args  = []
    i = 0
    for arg in args:
        call_args.append( arg.pname )
        if (arg.intent == 'in'):
            if (arg.array):
                # input arrays
                proto_args.append( '\n    ' + arg.dtype + ' const* ' + arg.name )
                query_args.append( arg.pname )
                if (arg.dtype in ('int64_t', 'bool')):
                    # integer input arrays: copy in input
                    local_vars += (tab + '#if 1\n'
                               +   tab*2 + '// 32-bit copy\n'
                               +   tab*2 + 'std::vector< blas_int > ' + arg.lname + '( &' + arg.name + '[0], &' + arg.name + '[' + arg.dim + '] );\n'
                               +   tab*2 + 'blas_int const* ' + arg.pname + ' = &' + arg.lname + '[0];\n'
                               +   tab + '#else\n'
                               +   tab*2 + 'blas_int const* ' + arg.pname + ' = ' + arg.lname + ';\n'
                               +   tab + '#endif\n')
                # end
            elif (arg.use_query):
                # lwork, lrwork, etc. local variables; not in proto_args
                query_args.append( '&ineg_one' )
            else:
                proto_args.append( arg.dtype + ' ' + arg.name )
                query_args.append( arg.pname )
                if (arg.dtype in ('int64_t', 'bool')):
                    # local 32-bit copy of 64-bit int
                    int_checks += tab*2 + 'throw_if_( std::abs(' + arg.name + ') > std::numeric_limits<blas_int>::max() );\n'
                    local_vars += tab + 'blas_int ' + arg.lname + ' = (blas_int) ' + arg.name + ';\n'
                else:
                    s = re.search( r'^lapack::(\w+)', arg.dtype )
                    if (s):
                        # local char copy of enum
                        enum = s.group(1).lower()
                        local_vars += tab + 'char ' + arg.lname + ' = ' + enum + '2char( ' + arg.name + ' );\n'
                # end
            # end
        else:
            if (arg.array and re.search( r'work', arg.name )):
                # work, rwork, etc. local variables; not in proto_args
                query_args.append( 'qry_' + arg.name )
                query += tab + arg.dtype + ' qry_' + arg.name + '[1];\n'

                if (arg.use_query):
                    alloc_work += tab + 'std::vector< ' + arg.dtype + ' > ' + arg.lname + '( ' + args[i+1].lname + ' );\n'
                else:
                    alloc_work += tab + 'std::vector< ' + arg.dtype + ' > ' + arg.lname + '( ' + arg.dim.lower() + ' );\n'

                ##alloc_work += (tab + arg.dtype + '* ' + arg.name + '_'
                ##           +   ' = new ' + arg.dtype + '[ l' + arg.name + ' ];\n')
                ##free_work  +=  tab + 'delete[] ' + arg.name + '_;\n'
            elif (not arg.array and arg.dtype in ('int64_t', 'bool')):
                # output dimensions (nfound, etc.)
                query_args.append( arg.pname )
                if (arg.name == 'info'):
                    # info not in proto_args
                    local_vars += tab + 'blas_int ' + arg.lname + ' = 0;\n'
                    retval = 'int64_t'
                    info_throw = (tab + 'if (info_ < 0) {\n'
                               +  tab*2 + 'throw Error();\n'
                               +  tab + '}\n')
                    info_return = tab + 'return info_;\n'
                else:
                    proto_args.append( '\n    ' + arg.dtype + '* ' + arg.name )
                    local_vars += tab + 'blas_int ' + arg.lname + ' = (blas_int) *' + arg.name + ';\n'
                    cleanup += tab + '*' + arg.name + ' = ' + arg.lname + ';\n'
            else:
                # output array
                query_args.append( arg.pname )
                proto_args.append( '\n    ' + arg.dtype + '* ' + arg.name )
                if (arg.array and (arg.dtype in ('int64_t', 'bool'))):
                    if (arg.intent == 'in,out'):
                        # copy in input, copy out in cleanup
                        local_vars += (tab + '#if 1\n'
                                   +   tab*2 + '// 32-bit copy\n'
                                   +   tab*2 + 'std::vector< blas_int > ' + arg.lname + '( &' + arg.name + '[0], &' + arg.name + '[' + arg.dim + '] );\n'
                                   +   tab*2 + 'blas_int* ' + arg.pname + ' = &' + arg.lname + '[0];\n'
                                   +   tab + '#else\n'
                                   +   tab*2 + 'blas_int* ' + arg.pname + ' = ' + arg.name + ';\n'
                                   +   tab + '#endif\n')
                    else:
                        # allocate w/o copy, copy out in cleanup
                        local_vars += (tab + '#if 1\n'
                                   +   tab*2 + '// 32-bit copy\n'
                                   +   tab*2 + 'std::vector< blas_int > ' + arg.lname + '( ' + arg.dim + ' );\n'
                                   +   tab*2 + 'blas_int* ' + arg.pname + ' = &' + arg.lname + '[0];\n'
                                   +   tab + '#else\n'
                                   +   tab*2 + 'blas_int* ' + arg.pname + ' = ' + arg.name + ';\n'
                                   +   tab + '#endif\n')
                    # end
                    cleanup += (tab + '#if 1\n'
                            +   tab*2 + 'std::copy( ' + arg.lname + '.begin(), ' + arg.lname + '.end(), ' + arg.name + ' );\n'
                            +   tab + '#endif\n')
            # end
        # end
        i += 1
    # end

    # --------------------
    # build query
    if (use_query):
        query =  ('\n'
              +   tab + '// query for workspace size\n'
              +   query
              +   tab + 'blas_int ineg_one = -1;\n'
              +   tab + 'LAPACK_' + func + '( ' + ', '.join( query_args ) + ' );\n'
              +   tab + 'if (info_ < 0) {\n'
              +   tab*2 + 'throw Error();\n'
              +   tab + '}\n')
        # assume when arg is l*work, last will be *work
        last = None
        for arg in args:
            if (arg.use_query and not arg.array):
                query += tab + 'blas_int ' + arg.name + '_ = real(qry_' + last.name + '[0]);\n'
            last = arg
        # end
    else:
        query = ''
    # end

    if (int_checks):
        int_checks = (tab + '// check for overflow\n'
                   +  tab + 'if (sizeof(int64_t) > sizeof(blas_int)) {\n'
                   +  int_checks
                   +  tab + '}\n')
    # end

    if (alloc_work):
        alloc_work = ('\n'
                   +  tab + '// allocate workspace\n'
                   +  alloc_work)
    # end

    if (is_func):
        call = '\n' + tab + 'return LAPACK_' + func + '( ' + ', '.join( call_args ) + ' );\n'
        if (info_throw or cleanup or info_return):
            print( 'Warning: why is info or cleanup on a function?' )
    else:
        call = '\n' + tab + 'LAPACK_' + func + '( ' + ', '.join( call_args ) + ' );\n'

    txt = ('// ' + '-'*77 + '\n'
        +  retval + ' ' + basefunc + '(\n'
        +  tab + ', '.join( proto_args )
        +  ' )\n{\n'
        +  int_checks
        +  local_vars
        +  query
        +  alloc_work
        +  call
        +  info_throw
        +  cleanup
        +  info_return
        +  '}\n')

    # trim trailing whitespace
    txt = re.sub( r' +$', r'', txt, 0, re.MULTILINE )
    return txt
# end


# ------------------------------------------------------------------------------
# Process each function on the command line, given as a base name like "getrf".
# Finds LAPACK functions with that base name and some precision prefix.
# Outputs to ../gen/basename.cc
lapack = os.environ['LAPACKDIR']

for arg in sys.argv[1:]:
    files = []
    for subdir in ('SRC', 'TESTING/MATGEN'):
        for p in ('s', 'd', 'c', 'z', 'ds', 'zc'):
            f = lapack + '/' + subdir + '/' + p + arg + ".f"
            if (os.path.exists( f )):
                files.append( f )
        # end
    # end

    if (not files):
        print( "no files found matching:", arg )
        continue
    # end

    f = "../gen/" + arg + ".cc"
    print( "generating", f )
    output = open( f, "w" )
    print( '''\
#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;
''', file=output )

    for f in files:
        print( '    ' + f )
        txt = process( f )
        print( txt, file=output )
    # end

    print( '}  // namespace lapack', file=output )
    output.close()

    print()
# end
