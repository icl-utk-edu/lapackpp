#!/usr/bin/env python

'''
wrapper_gen.py generates C++ wrappers, header prototypes, and testers for the
LAPACK Fortran routines by parsing doxygen comments in LAPACK's SRC *.f files.
It needs a copy of Netlib LAPACK source code, pointed to by $LAPACKDIR.
It takes a list of routines, without precision prefix (so "getrf" instead of "sgetrf", "dgetrf", ...).
It generates files in lapackpp/gen/ directory, which can then be manually moved to the appropriate place.

Arguments:
    -H, --header   generate header   in ../gen/lapack_wrappers.hh to go in ../include
    -w, --wrapper  generate wrappers in ../gen/foo.cc             to go in ../src
    -t, --tester   generate testers  in ../gen/test_foo.cc        to go in ../test

Example creating tester:
    slate> cd lapackpp/test

    # assume $LAPACKDIR is set
    slate/lapackpp/test> echo $LAPACKDIR
    /Users/mgates/Documents/lapack

    slate/lapackpp/test> ../tools/wrapper_gen.py -t posv
    generating ../gen/test_posv.cc
        /Users/mgates/Documents/lapack/SRC/sposv.f
        /Users/mgates/Documents/lapack/SRC/dposv.f
        /Users/mgates/Documents/lapack/SRC/cposv.f
        /Users/mgates/Documents/lapack/SRC/zposv.f

    slate/lapackpp/test> mv ../gen/test_posv.cc .
    slate/lapackpp/test> edit test.cc     # uncomment posv line
    slate/lapackpp/test> edit run_all.sh  # add posv line

    slate/lapackpp/test> make
    g++ ... -c -o test.o test.cc
    g++ ... -c -o test_posv.o test_posv.cc
    g++ ... -o test ...

    # initial version fails because matrix isn't positive definite
    slate/lapackpp/test> ./test posv
    input: ./test posv
                                               LAPACK++     LAPACK++     LAPACK++         Ref.         Ref.
      type    uplo       n    nrhs   align        error     time (s)      Gflop/s     time (s)      Gflop/s  status
    lapack::posv returned error 2
    LAPACKE_posv returned error 2
         d   lower     100      10       1   0.0000e+00       0.0000      20.7156       0.0000      14.2013  pass
    lapack::posv returned error 3
    LAPACKE_posv returned error 3
         d   lower     200      10       1   0.0000e+00       0.0001      23.7022       0.0001      27.6451  pass
    lapack::posv returned error 2
    LAPACKE_posv returned error 2
         d   lower     300      10       1   0.0000e+00       0.0001     135.7834       0.0002      65.2617  pass
    lapack::posv returned error 3
    LAPACKE_posv returned error 3
         d   lower     400      10       1   0.0000e+00       0.0001     300.1049       0.0002     121.8844  pass
    lapack::posv returned error 3
    LAPACKE_posv returned error 3
         d   lower     500      10       1   0.0000e+00       0.0001     676.7546       0.0002     195.6718  pass
    All tests passed.

    # add code to make matrix positive definite (e.g., diagonally dominant)
    slate/lapackpp/test> edit test_posv.cc
    slate/lapackpp/test> make
    g++ ... -c -o test_posv.o test_posv.cc
    g++ ... -o test ...

    # now tests pass; commit changes.
    slate/lapackpp/test> ./test posv --type s,d,c,z --dim 100:300:100
    input: ./test posv --type s,d,c,z --dim 100:300:100
                                               LAPACK++     LAPACK++     LAPACK++         Ref.         Ref.
      type    uplo       n    nrhs   align        error     time (s)      Gflop/s     time (s)      Gflop/s  status
         s   lower     100      10       1   0.0000e+00       0.0002       2.6913       0.0001       4.2364  pass
         s   lower     200      10       1   0.0000e+00       0.0005       6.4226       0.0004       7.9437  pass
         s   lower     300      10       1   0.0000e+00       0.0008      14.2683       0.0008      14.0091  pass

         d   lower     100      10       1   0.0000e+00       0.0002       2.7205       0.0001       3.8142  pass
         d   lower     200      10       1   0.0000e+00       0.0007       5.0169       0.0006       5.5775  pass
         d   lower     300      10       1   0.0000e+00       0.0009      11.4896       0.0009      11.7115  pass

         c   lower     100      10       1   0.0000e+00       0.0004       5.5265       0.0004       5.6153  pass
         c   lower     200      10       1   0.0000e+00       0.0011      12.7330       0.0012      11.9307  pass
         c   lower     300      10       1   0.0000e+00       0.0025      17.1538       0.0026      16.9923  pass

         z   lower     100      10       1   0.0000e+00       0.0005       4.4325       0.0005       4.4412  pass
         z   lower     200      10       1   0.0000e+00       0.0015       9.0585       0.0014      10.1714  pass
         z   lower     300      10       1   0.0000e+00       0.0035      12.3321       0.0032      13.7915  pass
    All tests passed.

    slate/lapackpp/test> hg commit -m 'test posv' test.cc test_posv.cc
'''

from __future__ import print_function

import sys
import re
import os
import argparse

from text_balanced import extract_bracketed

parser = argparse.ArgumentParser()
parser.add_argument( '-H', '--header',  help='generate header   in ../gen/lapack_wrappers.hh to go in ../include', action='store_true' )
parser.add_argument( '-w', '--wrapper', help='generate wrappers in ../gen/foo.cc             to go in ../src',     action='store_true' )
parser.add_argument( '-t', '--tester',  help='generate testers  in ../gen/test_foo.cc        to go in ../test',    action='store_true' )
parser.add_argument( 'argv', nargs=argparse.REMAINDER )
args = parser.parse_args()

# default
if (not (args.header or args.tester or args.wrapper)):
    args.wrapper = True


# ------------------------------------------------------------------------------
# configuration

debug = 0
#debug = 1

# --------------------
header_top = '''\
#ifndef ICL_LAPACK_WRAPPERS_HH
#define ICL_LAPACK_WRAPPERS_HH

#include "lapack_util.hh"

namespace lapack {

'''

header_bottom = '''\

}  // namespace lapack

#endif // ICL_LAPACK_WRAPPERS_HH
'''

# --------------------
wrapper_top = '''\
#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

'''

wrapper_bottom = '''\
}  // namespace lapack
'''

# --------------------
tester_top = '''\
#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

'''

tester_bottom = ''

# --------------------
typemap = {
    'character'        : 'char',
    'integer'          : 'int64_t',
    'real'             : 'float',
    'double precision' : 'double',
    'complex'          : 'std::complex<float>',
    'complex*16'       : 'std::complex<double>',
    'logical'          : 'bool',

    'logical function of two real arguments'               : 'lapack_s_select2',
    'logical function of three real arguments'             : 'lapack_s_select3',

    'logical function of two double precision arguments'   : 'lapack_d_select2',
    'logical function of three double precision arguments' : 'lapack_d_select3',

    'logical function of one complex argument'     : 'lapack_c_select1',
    'logical function of two complex arguments'    : 'lapack_c_select2',

    'logical function of one complex*16 argument'  : 'lapack_z_select1',
    'logical function of two complex*16 arguments' : 'lapack_z_select2',
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
    'balance': 'Balance',
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
        self.is_enum   = False
        self.is_array  = False
        self.dtype     = None
        self.dim       = None
        self.lbound    = None
        self.is_work   = False
        self.is_lwork  = False
    # end
# end

# ------------------------------------------------------------------------------
# captures information about each function
class Func:
    def __init__( self, xname ):
        self.xname   = xname
        self.is_func = False  # Fortran function or subroutine (default)
        self.args    = []
        self.retval  = 'void'

        # get base name of function (without precision)
        s = (re.search( '^(?:ds|zc)(gesv|posv)', xname ) or
             re.search( '^[sdcz](\w+)', xname ))
        if (s):
            self.name = s.group(1)
        else:
            self.name = 'unknown'
            print( 'unknown base function name', xname )
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
                lbound = join( lbound, 'max( ' + b + ' )' )
                found = True
            txt = r + s.group(2)
        else:
            s = re.search( arg.name + ' *>= *([^,;.]+)(.*)', txt, re.IGNORECASE | re.DOTALL )
            if (s):
                lbound = join( lbound, s.group(1) )
                found = True
                txt = s.group(2)
    # end
    lbound = re.sub( r',(\S)', r', \1', lbound )
    arg.lbound = lbound.lower()
# end

# ------------------------------------------------------------------------------
# state tracks what is looked for next:
# function, argument, argument's description, beginning with verbatim
state_func  = 0
state_arg   = 1
state_verb  = 2
state_desc  = 3

# ------------------------------------------------------------------------------
# Parses a single LAPACK src file, reading its function arguments.
# Returns a string with the wrapper.
def parse_lapack( path ):
    (d, filename) = os.path.split( path )
    (funcname, ext) = os.path.splitext( filename )
    func   = Func( funcname )
    state  = state_func
    arg    = None

    # --------------------
    # parse Fortran doxygen for arguments
    f = open( path )
    for line in f:
        if (state == state_func):
            s = re.search( r'^\* +\b(.*)\b +(?:recursive +)?function +(\w+)', line, re.IGNORECASE )  #\( *(.*) *\)', line )
            if (s):
                func.retval = typemap[ s.group(1).lower() ]
                func.is_func = True
                f2 = s.group(2).lower()
                if (func.xname != f2):
                    print( 'Warning: filename', filename, "doesn't match function", f2 )
                state = state_arg
            # end

            s = re.search( r'^\* +(?:recursive +)?subroutine +(\w+)', line, re.IGNORECASE )  #\( *(.*) *\)', line )
            if (s):
                f2 = s.group(1).lower()
                if (func.xname != f2):
                    print( 'Warning: filename', filename, "doesn't match function", f2 )
                state = state_arg
            # end
        elif (state == state_arg):
            s = re.search( r'^\*> *\\param\[(in|out|in,out|inout)\] +(\w+)', line )
            if (s):
                intent = s.group(1).lower()
                var    = s.group(2)
                arg = Arg( var, intent )
                func.args.append( arg )
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
    i = 0
    for arg in func.args:
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
                'TAU', 'TAUP', 'TAUQ', 'TAUA', 'TAUB',
                'ISUPPZ', 'IFAIL', 'ISEED', 'IBLOCK', 'ISPLIT',
                'SCALE', 'LSCALE', 'RSCALE',
                'ALPHA', 'ALPHAR', 'ALPHAI',
                'BETA', 'BETAR', 'BETAI',
                'THETA', 'PHI',
                'HOUS2',
                'ERR_BNDS_NORM', 'ERR_BNDS_COMP', 'PARAMS',
                'RCONDE', 'RCONDV')):
            arg.name = arg.name.lower()

        # mark work, lwork, ldwork, etc.
        if (re.search( '^[crsib]?work$', arg.name )):
            arg.is_work = True
        elif (re.search( '^(l|ld)[crsib]?work$', arg.name )):
            arg.is_lwork = True

        # iwork, bwork needs blas_int, not int64
        if (arg.name in ('iwork', 'bwork')):
            arg.dtype = 'blas_int'

        # extract array dimensions or scalar lower bounds
        if (arg.array):
            parse_dim( arg )
        else:
            parse_lbound( arg )

        # check for workspace query
        if (arg.is_lwork):
            s = re.search( 'If ' + arg.name + ' *= *-1.*workspace query', arg.desc, re.DOTALL | re.IGNORECASE )
            if (s):
                func.args[i-1].use_query = True  # work
                arg.use_query = True  # lwork
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
        elif (arg.is_work):
            arg.pname = '&' + arg.name + '[0]'
        # end

        # map char to enum (after doing lname)
        if (arg.dtype == 'char'):
            arg.is_enum = True
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
    return func
# end

# ------------------------------------------------------------------------------
def generate_wrapper( func, header=False ):
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
    use_query  = False
    i = 0
    for arg in func.args:
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
                use_query = True
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
            if (arg.array and arg.is_work):
                # work, rwork, etc. local variables; not in proto_args
                query_args.append( 'qry_' + arg.name )
                query += tab + arg.dtype + ' qry_' + arg.name + '[1];\n'

                if (arg.use_query):
                    alloc_work += tab + 'std::vector< ' + arg.dtype + ' > ' + arg.lname + '( ' + func.args[i+1].lname + ' );\n'
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
                    func.retval = 'int64_t'
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
              +   tab + 'LAPACK_' + func.xname + '( ' + ', '.join( query_args ) + ' );\n'
              +   tab + 'if (info_ < 0) {\n'
              +   tab*2 + 'throw Error();\n'
              +   tab + '}\n')
        # assume when arg is l*work, last will be *work
        last = None
        for arg in func.args:
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

    if (func.is_func):
        call = '\n' + tab + 'return LAPACK_' + func.xname + '( ' + ', '.join( call_args ) + ' );\n'
        if (info_throw or cleanup or info_return):
            print( 'Warning: why is info or cleanup on a function?' )
    else:
        call = '\n' + tab + 'LAPACK_' + func.xname + '( ' + ', '.join( call_args ) + ' );\n'

    if (header):
        txt = (func.retval + ' ' + func.name + '(\n'
            +  tab + ', '.join( proto_args )
            +  ' );\n\n')
    else:
        txt = ('// ' + '-'*77 + '\n'
            +  func.retval + ' ' + func.name + '(\n'
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
            +  '}\n\n')
    # end

    # trim trailing whitespace
    txt = re.sub( r' +$', r'', txt, 0, re.MULTILINE )
    return txt
# end

# ------------------------------------------------------------------------------
def generate_tester( funcs ):
    # heuristic:
    # assume ints before 1st array are input parameters (trans, m, n, ...)
    # while  ints after  1st array are derived (lda, ...)
    pre_arrays = True

    scalars  = ''  # pre-arrays
    scalars2 = ''  # post-arrays
    sizes    = ''
    arrays   = ''
    init     = (tab + 'int64_t idist = 1;\n'
             +  tab + 'int64_t iseed[4] = { 0, 1, 2, 3 };\n')
    copy     = ''
    flop_args = [] # arguments for calling Gflops::foo(), pre-array
    tst_args = []  # arguments for calling C++ test
    ref_args = []  # arguments for calling Fortran reference
    verify   = ''

    func = funcs[-1]  # guess last one is complex, to get real_t below
    for arg in func.args:
        if (not (arg.is_work or arg.is_lwork or arg.name == 'info')):
            # determine template type from data type
            if (arg.dtype in ('float', 'double')):
                if (re.search( r'^[cz]', func.xname )):
                    arg.ttype = 'real_t'
                else:
                    arg.ttype = 'scalar_t'
                arg.ttype_ref = arg.ttype
            elif (arg.dtype in ('std::complex<float>', 'std::complex<double>')):
                arg.ttype = 'scalar_t'
                arg.ttype_ref = arg.ttype
            elif (arg.dtype in ('int64_t')):
                arg.ttype = arg.dtype
                arg.ttype_ref = 'lapack_int'
            else:
                arg.ttype = arg.dtype
                arg.ttype_ref = arg.ttype

            if (arg.array):
                pre_arrays = False

                # look for 2-D arrays: (m,n), change to: m * n
                s = re.search( r'^ *\( *(\w+), *(\w+) *\) *$', arg.dim )
                if (s):
                    dim = s.group(1) + ' * ' + s.group(2)
                else:
                    dim = arg.dim

                sizes += tab + 'size_t size_' + arg.name + ' = (size_t) ' + dim + ';\n';
                if (arg.intent == 'in' and not arg.dtype in ('int64_t')):
                    arrays += tab + 'std::vector< ' + arg.ttype + ' > ' + arg.name + '( size_' + arg.name + ' );\n'
                    if (arg.dtype in ('float', 'double', 'std::complex<float>', 'std::complex<double>')):
                        init += tab + 'lapack::larnv( idist, iseed, ' + arg.name + '.size(), &' + arg.name + '[0] );\n'
                    else:
                        init += tab + '// todo: initialize ' + arg.name
                    tst_args.append( '&' + arg.name + '[0]' )
                    ref_args.append( '&' + arg.name + '[0]' )
                else:
                    arrays += (tab + 'std::vector< ' + arg.ttype     + ' > ' + arg.name + '_tst( size_' + arg.name + ' );\n'
                           +   tab + 'std::vector< ' + arg.ttype_ref + ' > ' + arg.name + '_ref( size_' + arg.name + ' );\n')
                    if ('in' in arg.intent):
                        if (arg.dtype in ('float', 'double', 'std::complex<float>', 'std::complex<double>')):
                            init += tab + 'lapack::larnv( idist, iseed, ' + arg.name + '_tst.size(), &' + arg.name + '_tst[0] );\n'
                            copy += tab + arg.name + '_ref = ' + arg.name + '_tst;\n'
                        elif ('in' in arg.intent):
                            init += tab + '// todo: initialize ' + arg.name + '_tst and ' + arg.name + '_ref\n'
                    # end
                    tst_args.append( '&' + arg.name + '_tst[0]' )
                    ref_args.append( '&' + arg.name + '_ref[0]' )
                    if ('out' in arg.intent):
                        verify += tab*2 + 'error += abs_error( ' + arg.name + '_tst, ' + arg.name + '_ref );\n'
                # end
            else:
                # not array
                if (arg.intent == 'in'):
                    if (pre_arrays):
                        if (arg.name not in ('uplo')):
                            flop_args.append( arg.name )
                        if (arg.name in ('m', 'n', 'k')):
                            scalars += tab + arg.ttype + ' ' + arg.name + ' = params.dim.' + arg.name + '();\n'
                        else:
                            scalars += tab + arg.ttype + ' ' + arg.name + ' = params.' + arg.name + '.value();\n'
                    elif (arg.lbound):
                        if (arg.name.startswith('ld')):
                            scalars2 += tab + arg.ttype + ' ' + arg.name + ' = roundup( ' + arg.lbound + ', align );\n'
                        else:
                            scalars2 += tab + arg.ttype + ' ' + arg.name + ' = ' + arg.lbound + ';\n'
                    elif ('in' in arg.intent):
                        scalars2 += tab + arg.ttype + ' ' + arg.name + ';  // todo value\n'
                    else:
                        scalars2 += tab + arg.ttype + ' ' + arg.name + ';\n'
                    tst_args.append( arg.name )
                    if (arg.is_enum):
                        # convert enum to char, e.g., uplo2char(uplo)
                        s = re.search( '^lapack::(\w+)', arg.dtype )
                        assert( s is not None )
                        ref_args.append( s.group(1).lower() + '2char(' + arg.name + ')' )
                    else:
                        ref_args.append( arg.name )
                else:
                    if (pre_arrays):
                        scalars += tab + arg.ttype + ' ' + arg.name + '_tst = params.' + arg.name + '.value();\n'
                        scalars += tab + arg.ttype + ' ' + arg.name + '_ref = params.' + arg.name + '.value();\n'
                    elif (arg.lbound):
                        scalars2 += tab + arg.ttype + ' ' + arg.name + '_tst = ' + arg.lbound + ';\n'
                        scalars2 += tab + arg.ttype + ' ' + arg.name + '_ref = ' + arg.lbound + ';\n'
                    elif ('in' in arg.intent):
                        scalars2 += tab + arg.ttype + ' ' + arg.name + '_tst;  // todo value\n'
                        scalars2 += tab + arg.ttype + ' ' + arg.name + '_ref;  // todo value\n'
                    else:
                        scalars2 += tab + arg.ttype + ' ' + arg.name + '_tst;\n'
                        scalars2 += tab + arg.ttype + ' ' + arg.name + '_ref;\n'
                    tst_args.append( '&' + arg.name + '_tst' )
                    ref_args.append( '&' + arg.name + '_ref' )
                    verify += tab*2 + 'error += std::abs( ' + arg.name + '_tst - ' + arg.name + '_ref );\n'
                # end
            # end
        # end
    # end

    lapacke  = ('// -----------------------------------------------------------------------------\n'
             +  '// simple overloaded wrappers around LAPACKE\n')
    for func in funcs:
        lapacke_proto = []
        lapacke_args  = []
        for arg in func.args:
            if (arg.name in ('info', 'work', 'rwork', 'iwork', 'lwork', 'lrwork', 'ldwork', 'liwork')):
                continue
            elif (arg.is_enum):
                lapacke_proto.append( 'char ' + arg.name )
            elif (arg.dtype in ('int64_t')):
                if (arg.array):
                    lapacke_proto.append( 'lapack_int* ' + arg.name )
                else:
                    lapacke_proto.append( 'lapack_int ' + arg.name )
            else:
                if (arg.array or ('out' in arg.intent)):
                    lapacke_proto.append( arg.dtype + '* ' + arg.name )
                else:
                    lapacke_proto.append( arg.dtype + ' ' + arg.name )
            # end
            lapacke_args.append( arg.name )
        # end
        lapacke_proto = ', '.join( lapacke_proto )
        lapacke_args  = ', '.join( lapacke_args )
        lapacke += ('static lapack_int LAPACKE_' + func.name + '(\n'
                +   tab + lapacke_proto + ' )\n'
                +   '{\n'
                +   tab + 'return LAPACKE_' + func.xname + '( LAPACK_COL_MAJOR, ' + lapacke_args + ' );\n'
                +   '}\n\n')
    # end

    flop_args = ', '.join( flop_args )
    tst_args = ', '.join( tst_args )
    ref_args = ', '.join( ref_args )

    # todo: only dispatch for precisions in funcs
    dispatch = ('''
// -----------------------------------------------------------------------------
void test_''' + func.name + '''( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_''' + func.name + '''_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_''' + func.name + '''_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_''' + func.name + '''_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_''' + func.name + '''_work< std::complex<double> >( params, run );
            break;
    }
}
''')

    txt = (lapacke
        +  '// -----------------------------------------------------------------------------\n'
        +  'template< typename scalar_t >\n'
        +  'void test_' + func.name + '_work( Params& params, bool run )\n'
        +  '{\n'
        +  tab + 'using namespace blas;\n'
        +  tab + 'typedef typename traits< scalar_t >::real_t real_t;\n'
        +  tab + 'typedef long long lld;\n'
        +  '\n'
        +  tab + '// get & mark input values\n'
        +  scalars
        +  tab + 'int64_t align = params.align.value();\n'
        #+ tab + 'int64_t verbose = params.verbose.value();\n'
        +  '\n'
        +  tab + '// mark non-standard output values\n'
        +  tab + 'params.ref_time.value();\n'
        +  tab + 'params.ref_gflops.value();\n'
        +  tab + 'params.gflops.value();\n'
        +  '\n'
        +  tab + 'if (! run)\n'
        +  tab*2 + 'return;\n'
        +  '\n'
        +  tab + '// ---------- setup\n'
        +  scalars2
        +  sizes
        +  '\n'
        +  arrays
        +  '\n'
        +  init
        +  copy
        +  '\n'
        +  tab + '// ---------- run test\n'
        +  tab + 'libtest::flush_cache( params.cache.value() );\n'
        +  tab + 'double time = omp_get_wtime();\n'
        +  tab + 'int64_t info_tst = lapack::' + func.name + '( ' + tst_args + ' );\n'
        +  tab + 'time = omp_get_wtime() - time;\n'
        +  tab + 'if (info_tst != 0) {\n'
        +  tab + '    fprintf( stderr, "lapack::' + func.name + ' returned error %lld\\n", (lld) info_tst );\n'
        +  tab + '}\n'
        +  '\n'
        +  tab + 'double gflop = lapack::Gflop< scalar_t >::' + func.name + '( ' + flop_args + ' );\n'
        +  tab + 'params.time.value()   = time;\n'
        +  tab + 'params.gflops.value() = gflop / time;\n'
        +  '\n'
        +  tab + "if (params.ref.value() == 'y' || params.check.value() == 'y') {\n"
        +  tab*2 + '// ---------- run reference\n'
        +  tab*2 + 'libtest::flush_cache( params.cache.value() );\n'
        +  tab*2 + 'time = omp_get_wtime();\n'
        +  tab*2 + 'int64_t info_ref = LAPACKE_'  + func.name + '( ' + ref_args + ' );\n'
        +  tab*2 + 'time = omp_get_wtime() - time;\n'
        +  tab*2 + 'if (info_ref != 0) {\n'
        +  tab*2 + '    fprintf( stderr, "LAPACKE_' + func.name + ' returned error %lld\\n", (lld) info_ref );\n'
        +  tab*2 + '}\n'
        +  '\n'
        +  tab*2 + 'params.ref_time.value()   = time;\n'
        +  tab*2 + 'params.ref_gflops.value() = gflop / time;\n'
        +  '\n'
        +  tab*2 + '// ---------- check error compared to reference\n'
        +  tab*2 + 'real_t error = 0;\n'
        +  tab*2 + 'if (info_tst != info_ref) {\n'
        +  tab*2 + '    error = 1;\n'
        +  tab*2 + '}\n'
        +  verify
        +  tab*2 + 'params.error.value() = error;\n'
        +  tab*2 + 'params.okay.value() = (error == 0);  // expect lapackpp == lapacke\n'
        +  tab + '}\n'
        +  '}\n'
        +  dispatch
    )

    # trim trailing whitespace
    txt = re.sub( r' +$', r'', txt, 0, re.MULTILINE )
    return txt
# end

# ------------------------------------------------------------------------------
# Process each function on the command line, given as a base name like "getrf".
# Finds LAPACK functions with that base name and some precision prefix.
# Outputs to ../gen/basename.cc
lapack = os.environ['LAPACKDIR']

if (args.header):
    header_file = '../gen/lapack_wrappers.hh'
    print( 'generating', header_file )
    header = open( header_file, 'w' )
    print( header_top, file=header, end='' )

for arg in args.argv:
    files = []
    for subdir in ('SRC', 'TESTING/MATGEN'):
        for p in ('s', 'd', 'c', 'z'):  #, 'ds', 'zc'):
            f = lapack + '/' + subdir + '/' + p + arg + ".f"
            if (os.path.exists( f )):
                files.append( f )
        # end
    # end

    if (not files):
        print( "no files found matching:", arg )
        continue
    # end

    if (args.header):
        print( '// ' + '-'*77, file=header )

    if (args.wrapper):
        wrapper_file = '../gen/' + arg + '.cc'
        print( 'generating', wrapper_file )
        wrapper = open( wrapper_file, 'w' )
        print( wrapper_top, file=wrapper, end='' )

    if (args.tester):
        tester_file = '../gen/test_' + arg + '.cc'
        print( 'generating', tester_file )
        tester = open( tester_file, 'w' )
        print( tester_top, file=tester, end='' )

    funcs = []
    for f in files:
        print( '    ' + f )
        func = parse_lapack( f )
        funcs.append( func )
        if (args.header):
            txt = generate_wrapper( func, header=True )
            print( txt, file=header, end='' )
        if (args.wrapper):
            txt = generate_wrapper( func )
            print( txt, file=wrapper, end='' )
    # end

    # tester needs to see all related functions
    if (args.tester):
        txt = generate_tester( funcs )
        print( txt, file=tester, end='' )

    if (args.wrapper):
        print( wrapper_bottom, file=wrapper, end='' )
        wrapper.close()

    if (args.tester):
        print( tester_bottom, file=tester, end='' )
        tester.close()

    print()
# end

if (args.header):
    print( header_bottom, file=header, end='' )
    header.close()
