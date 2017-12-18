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

alias_map = {
    'sy': 'he',
    'sp': 'hp',
    'sb': 'hb',
    'or': 'un',
    'op': 'up',
}

# 4 space indent
tab = '    '


# ------------------------------------------------------------------------------
# captures information about each function argument
class Arg:
    def __init__( self, name, intent ):
        self.name      = name
        self.name_orig = name
        self.intent    = intent
        self.desc      = ''

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
        self.purpose = ''
        self.details = ''
        self.group   = 'unknown'

        # get base name of function (without precision)
        s = (re.search( r'^(?:ds|zc)(gesv|posv)', xname ) or
             re.search( r'^[sdcz](\w+)', xname ))
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
# Sets arg.dim to string of all dimensions, seperated by ";".
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
#   function,
#   purpose,
#   param, "verbatim", param's description [repeated],
#   ingroup
i = 0
state_func          = i; i += 1

state_purpose       = i; i += 1
state_purpose_verb  = i; i += 1
state_purpose_desc  = i; i += 1

state_param         = i; i += 1
state_param_verb    = i; i += 1
state_param_desc    = i; i += 1

state_details       = i; i += 1
state_details_verb  = i; i += 1
state_details_desc  = i; i += 1

state_done          = i; i += 1

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
    # parse Fortran doxygen for purpose and params
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
                state = state_purpose
                continue
            # end

            s = re.search( r'^\* +(?:recursive +)?subroutine +(\w+)', line, re.IGNORECASE )  #\( *(.*) *\)', line )
            if (s):
                f2 = s.group(1).lower()
                if (func.xname != f2):
                    print( 'Warning: filename', filename, "doesn't match function", f2 )
                state = state_purpose
            # end

        # ----------
        elif (state == state_purpose):
            if (re.search( r'^\*> *\\par +Purpose:', line )):
                state = state_purpose_verb

        elif (state == state_purpose_verb):
            if (re.search( r'^\*> *\\verbatim', line )):
                state = state_purpose_desc

        elif (state == state_purpose_desc):
            if (re.search( r'^\*> *\\endverbatim', line )):
                state = state_param
            else:
                func.purpose += line

        # ----------
        elif (state == state_param):
            s = re.search( r'^\*> *\\param\[(in|out|in,out|inout)\] +(\w+)', line )
            if (s):
                intent = s.group(1).lower()
                var    = s.group(2)
                arg = Arg( var, intent )
                func.args.append( arg )
                state = state_param_verb
                continue
            # end

            # get group minus precision (sdcz prefix)
            s = re.search( r'^\*> *\\ingroup +(?:[sdcz](\w+)|(\w+))', line )
            if (s):
                func.group = s.group(1) or s.group(2)
                state = state_details
                #print( 'details head (1)' )
                continue

            # shouldn't happen, but if \\ingroup isn't before Further Details
            if (re.search( r'^\*> *\\par Further Details', line )):
                state = state_further_verb
                #print( 'details head (1)' )
                continue

            if (re.search( r'^      SUBROUTINE', line )):
                # finished!
                #print( 'finished (1)' )
                break

        elif (state == state_param_verb):
            if (re.search( r'^\*> *\\verbatim', line )):
                state = state_param_desc

        elif (state == state_param_desc):
            if (re.search( r'^\*> *\\endverbatim', line )):
                state = state_param
                arg = None
            else:
                arg.desc += line

        # ----------
        elif (state == state_details):
            #print( line )
            if (re.search( r'^\*> *\\par Further Details', line )):
                state = state_details_verb
                #print( 'details head (2)' )
            elif (re.search( r'^      SUBROUTINE', line )):
                # finished!
                #print( 'finished (2)' )
                break

        elif (state == state_details_verb):
            if (re.search( r'^\*> *\\verbatim', line )):
                state = state_details_desc
                #func.details += '### Further Details\n'
                #print( 'details verb' )

        elif (state == state_details_desc):
            if (re.search( r'^\*> *\\endverbatim', line )):
                # finished!
                #print( 'finished (3)' )
                break
            else:
                func.details += line
        # end
    # end

    # strip comment chars; squeeze space
    func.purpose = re.sub( r'\*> ?',   r'',    func.purpose )
    func.purpose = re.sub( r'(\S)  +', r'\1 ', func.purpose )

    # strip comment chars
    # LAPACK details tend to be indented 2 spaces instead of 1 space
    func.details = re.sub( r'\*> {0,2}', r'', func.details )
    # for now, don't squeeze space (destroys examples like in gebrd)
    #func.details = re.sub( r'(\S)  +', r'\1 ', func.details )

    # --------------------
    # parse arguments for properties
    i = 0
    for arg in func.args:
        if (debug): print( '-'*40 )

        # strip comment chars; squeeze space
        # normally, lapack indents 10 spaces; if more, leave excess
        arg.desc = re.sub( r'\*> {0,10}', r'',    arg.desc )
        arg.desc = re.sub( r'(\S)  +',    r'\1 ', arg.desc )

        # extract data type and if array
        s = (re.search( r'^ *' + arg.name + ' +is +(CHARACTER|INTEGER|REAL|DOUBLE PRECISION|COMPLEX\*16|COMPLEX|LOGICAL)( array)?', arg.desc ) or
             re.search( r'^ *\(workspace\) +(CHARACTER|INTEGER|REAL|DOUBLE PRECISION|COMPLEX\*16|COMPLEX|LOGICAL)( array)?', arg.desc ) or
             re.search( r'^ *' + arg.name + ' +is a (LOGICAL FUNCTION of (?:one|two|three) (?:REAL|DOUBLE PRECISION|COMPLEX\*16|COMPLEX) arguments?)( array)?', arg.desc ))
        if (s):
            arg.dtype = typemap[ s.group(1).lower() ]
            arg.is_array = (s.group(2) == ' array')
        else:
            print( 'ERROR: unknown dtype:\n' + arg.desc )

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
        if (not arg.is_array or arg.name in
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
        if (re.search( r'^[crsib]?work$', arg.name )):
            arg.is_work = True
        elif (re.search( r'^(l|ld)[crsib]?work$', arg.name )):
            arg.is_lwork = True

        # iwork, bwork needs blas_int, not int64
        if (arg.name in ('iwork', 'bwork')):
            arg.dtype = 'blas_int'

        # extract array dimensions or scalar lower bounds
        if (arg.is_array):
            parse_dim( arg )
        else:
            parse_lbound( arg )

        # check for workspace query
        if (arg.is_lwork):
            s = re.search( r'If ' + arg.name + ' *= *-1.*workspace query', arg.desc, re.DOTALL | re.IGNORECASE )
            if (s):
                func.args[i-1].use_query = True  # work
                arg.use_query = True  # lwork
        # end

        # lname = local name
        # pname = pointer name to pass to Fortran
        arg.lname = arg.name
        arg.pname = arg.name
        if (arg.intent == 'in' and not arg.is_array and not re.search( r'LAPACK_._SELECT', arg.dtype)):
            # scalars, dimensions, enums
            if (arg.dtype in ('int64_t', 'bool', 'char')):
                # dimensions, enums
                arg.lname += '_'
            arg.pname = '&' + arg.lname
        elif (arg.intent in ('out', 'in,out') and not arg.is_array and arg.dtype in ('int64_t', 'bool')):
            # output dimensions (nfound, rank, ...)
            arg.lname += '_'
            arg.pname = '&' + arg.lname
        elif (arg.is_array and arg.dtype in ('int64_t', 'bool')):
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
            if (arg.is_array):
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
enums = {
    'uplo': {
        'u': ('Upper', 'lapack::Uplo::Upper'),
        'l': ('Lower', 'lapack::Uplo::Lower'),
    },
    'trans': {
        'n': ('NoTrans',   'lapack::Op::NoTrans'),
        't': ('Trans',     'lapack::Op::Trans'),
        'c': ('ConjTrans', 'lapack::Op::ConjTrans'),
    },
    'side': {
        'l': ('Left',  'lapack::Side::Left'),
        'r': ('Right', 'lapack::Side::Right'),
    },
    'norm': {
        '1': ('One', 'lapack::Norm::One'),
        'o': ('One', 'lapack::Norm::One'),
        'i': ('Inf', 'lapack::Norm::Inf'),
        'f': ('Fro', 'lapack::Norm::Fro'),
        'm': ('Max', 'lapack::Norm::Max'),
    },
    'diag': {
        'n': ('NonUnit', 'lapack::Diag::NonUnit'),
        'u': ('Unit',    'lapack::Diag::Unit'),
    },
    'compq': {
        'n': ('NoVec',      'lapack::CompQ::NoVec'),
        'i': ('Vec',        'lapack::CompQ::Vec'),
        'p': ('CompactVec', 'lapack::CompQ::CompactVec'),
        'v': ('Update',     'lapack::CompQ::Update'),
    },
    'vect': {
        'q': ('Q',    'lapack::Vect::Q'    ),
        'p': ('P',    'lapack::Vect::P'    ),
        'n': ('None', 'lapack::Vect::None' ),
        'b': ('Both', 'lapack::Vect::Both' ),
    },
    'equed': {
        'r': ('Row',  'lapack::Equed::Row'),
        'c': ('Col',  'lapack::Equed::Col'),
        'n': ('None', 'lapack::Equed::None'),
        'b': ('Both', 'lapack::Equed::Both'),
    },
    'fact': {
        'f': ('Factored',    'lapack::Factored::Factored'),
        'n': ('NotFactored', 'lapack::Factored::NotFactored'),
        'e': ('Equilibrate', 'lapack::Factored::Equilibrate'),
    },

    # geev
    'jobvl': {
        'n': ('NoVec',        'lapack::Job::NoVec'       ),
        'v': ('Vec',          'lapack::Job::Vec'         ),
    },
    'jobvr': {
        'n': ('NoVec',        'lapack::Job::NoVec'       ),
        'v': ('Vec',          'lapack::Job::Vec'         ),
    },

    # gesvd
    'jobu': {
        'n': ('NoVec',        'lapack::Job::NoVec'       ),
        'a': ('AllVec',       'lapack::Job::AllVec'      ),
        's': ('SomeVec',      'lapack::Job::SomeVec'     ),
        'o': ('OverwriteVec', 'lapack::Job::OverwriteVec'),
    },
    'jobvt': {
        'n': ('NoVec',        'lapack::Job::NoVec'       ),
        'a': ('AllVec',       'lapack::Job::AllVec'      ),
        's': ('SomeVec',      'lapack::Job::SomeVec'     ),
        'o': ('OverwriteVec', 'lapack::Job::OverwriteVec'),
    },

    # bbcsd
    'jobu1': {
        'n': ('NoUpdate', 'lapack::JobCS::NoUpdate'),
        'y': ('Update',   'lapack::JobCS::Update'),
    },
    'jobu2': {
        'n': ('NoUpdate', 'lapack::JobCS::NoUpdate'),
        'y': ('Update',   'lapack::JobCS::Update'),
    },
    'jobv1t': {
        'n': ('NoUpdate', 'lapack::JobCS::NoUpdate'),
        'y': ('Update',   'lapack::JobCS::Update'),
    },
    'jobv2t': {
        'n': ('NoUpdate', 'lapack::JobCS::NoUpdate'),
        'y': ('Update',   'lapack::JobCS::Update'),
    },

    # gesdd, syev
    'jobz': {
        'n': ('NoVec',        'lapack::Job::NoVec'       ),
        'v': ('Vec',          'lapack::Job::Vec'         ),
        'a': ('AllVec',       'lapack::Job::AllVec'      ),
        's': ('SomeVec',      'lapack::Job::SomeVec'     ),
        'o': ('OverwriteVec', 'lapack::Job::OverwriteVec'),
    },
}

# ------------------------------------------------------------------------------
def sub_enum_short( match ):
    groups = match.groups()
    try:
        #pre  = groups[0]
        enum = groups[0].lower()
        mid  = groups[1]
        val  = groups[2].lower()
        txt  = enum + mid + enums[enum][val][0]
        if (len(groups) > 3 and groups[4]):
            mid2 = groups[3]
            val2 = groups[4].lower()
            if (enums[enum][val][0] != enums[enum][val2][0]):
                txt += mid2 + enums[enum][val2][0]
        return txt
    except:
        print( 'unknown enum', enum, 'groups', groups )
        return match.group(0) + ' [TODO: unknown enum]'
# end

# ------------------------------------------------------------------------------
fortran_operators = {
    'LT': '<',
    'GT': '>',
    'LE': '<=',
    'GE': '>=',
    'EQ': '==',
    'NE': '!=',
}

# ------------------------------------------------------------------------------
def parse_docs( txt, variables, indent=0 ):
    # exponents
    txt = re.sub( r'\*\*', r'^', txt )

    # Fortran operators:  ".LT."  =>  "<"
    txt = re.sub( r'\.(LT|LE|GT|GE|EQ|NE)\.',
                  lambda match: fortran_operators[ match.group(1) ], txt )

    # space around operators
    txt = re.sub( r'(\S)(<=|>=)(\S)', r'\1 \2 \3', txt )
    txt = re.sub( r'(\S)(<=|>=)(\S)', r'\1 \2 \3', txt )

    # make indents 4 spaces (lapack usually has 3)
    txt = re.sub( r'^   +', r'    ', txt, 0, re.M )

    # convert enums, e.g.,
    # "UPLO = 'U'" => "uplo = Upper"
    txt = re.sub( r"\b([A-Z]{2,})( *= *)'(\w)'(?:( or )'(\w)')?", sub_enum_short, txt )

    # convert function names
    txt = re.sub( r'((?:computed\s+by|determined\s+by|from)\s+)[SDCZ]([A-Z_]{4,})\b',
                  lambda match: match.group(1) + '`lapack::' + match.group(2).lower() + '`',
                  txt )
    #txt = re.sub( r'(from\s+)[SDCZ]([A-Z]{4,})\b',
    #              lambda match: match.group(1) + '`lapack::' + match.group(2).lower() + '`',
    #              txt )
    txt = re.sub( func.xname.upper(), '`'+ func.name + '`', txt )

    # INFO references
    txt = re.sub( r'On exit, if INFO *= *0,', r'On successful exit,', txt, 0, re.I )
    txt = re.sub( r'If INFO *= *0', r'If successful', txt, 0, re.I )
    txt = re.sub( r'\bINFO *(=|>|>=) *', r'return value \1 ', txt )

    # rename arguments (lowercase, spelling)
    for (search, replace) in variables:
        txt = re.sub( search, replace, txt )

    # hyphenate m-by-n
    txt = re.sub( r'\b([a-z]) by ([a-z])\b', r'\1-by-\2', txt )

    # prefix lines
    txt = re.sub( r'^', r'/// ' + ' '*indent, txt, 0, re.M ) + '\n'
    return txt
# end

# ------------------------------------------------------------------------------
def generate_docs( func, header=False ):
    variables = []
    for arg in func.args:
        if (arg.name_orig != arg.name):
            variables.append( (r'\b' + arg.name_orig + r'\b', arg.name) )
    #print( 'variables', variables )

    txt = ''

    # --------------------
    d = func.purpose

    # strip function name and capitalize first word, e.g.,
    # "ZPOTRF computes ..." => "Computes ..."
    d = re.sub( r'^\s*[A-Z_]+ (\w)', lambda s: s.group(1).upper(), d )
    txt += parse_docs( d, variables, indent=0 )
    txt += '''\
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
'''

    # --------------------
    i = 0
    for arg in func.args:
        if (arg.is_work or arg.is_lwork):
            continue

        d = arg.desc

        # remove "VAR is INTEGER" line
        d = re.sub( r'^ *' + arg.name.upper() + ' +is +(CHARACTER|INTEGER|REAL|DOUBLE PRECISION|COMPLEX\*16|COMPLEX|LOGICAL).*\n', r'', d )

        if (arg.name == 'info'):
            # @retval
            d = re.sub( r'^ *< *0: *if INFO *= *-i, the i-th argument.*\n', r'', d, 0, re.M | re.I )
            d = re.sub( r'^ *(=|<|<=|>|>=) ', r'@retval \1 ', d, 0, re.M )

            # increase indented lines
            d = re.sub( r'^      ', r'             ', d, 0, re.M )

            txt += parse_docs( d, variables, indent=4 )
        else:
            txt += '/// @param[' + arg.intent + '] ' + arg.name + '\n'

            if (arg.is_array):
                s = re.search( r'^ *\( *([^(),]+), *([^(),]+) *\) *$', arg.dim )
                if (s):
                    rows = 'ROWS'
                    ld   = s.group(1)
                    cols = s.group(2)
                    dim = ld + '-by-' + cols
                    if (ld.startswith('ld') and i+1 < len(func.args)):
                        ld_arg = func.args[i+1]
                        if (ld_arg.lbound):
                            s = re.search( r'^ *max\( *1, *(\w+) *\)', ld_arg.lbound )
                            if (s):
                                rows = s.group(1)
                            else:
                                rows = ld_arg.lbound
                    # end
                    dim2 = rows + '-by-' + cols
                    d = 'The ' + dim2 + ' matrix ' + arg.name + ', stored in an ' \
                      + dim + ' array.\n' + d
                    # [dimension ' + arg.dim + ']
                else:
                    # try to strip off one set of parens
                    (dim, rem) = extract_bracketed( arg.dim, '(', ')' )
                    if (txt and rem == ''):
                        d = 'The vector ' + arg.name + ' of length ' + dim + '.\n' + d
                    else:
                        d = 'The vector ' + arg.name + ' of length ' + arg.dim + '.\n' + d
            # end

            # convert enum values, e.g.,
            # = 'U': ...    =>    - Upper: ...
            # = 'L': ...    =>    - Lower: ...
            if (arg.is_enum):
                d = re.sub( r"^ *= *'(\w)'( or '\w')?:",
                            lambda match: '- ' + enums[arg.name][match.group(1).lower()][1] + ':',
                            d, 0, re.M )
            # end

            d = parse_docs( d, variables, indent=4 )

            # add "\n" on blank lines
            d = re.sub( r'^(/// +)\n///', r'\1\\n\n///', d, 0, re.M )
            txt += d
        # end
        i += 1
    # end

    # Further Details section
    d = func.details
    if (d):
        txt += '// -----------------------------------------------------------------------------\n'
        txt += '/// @par Further Details\n'
        txt += parse_docs( d, variables, indent=4 )
    # end

    txt += '/// @ingroup ' + func.group + '\n'

    # trim trailing whitespace
    txt = re.sub( r' +$', r'', txt, 0, re.M )
    return txt
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
    alias_args = []
    use_query  = False
    i = 0
    for arg in func.args:
        call_args.append( arg.pname )
        if (arg.intent == 'in'):
            if (arg.is_array):
                # input arrays
                proto_args.append( '\n    ' + arg.dtype + ' const* ' + arg.name )
                alias_args.append( arg.name )
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
                alias_args.append( arg.name )
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
            if (arg.is_array and arg.is_work):
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
            elif (not arg.is_array and arg.dtype in ('int64_t', 'bool')):
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
                    alias_args.append( arg.name )
                    local_vars += tab + 'blas_int ' + arg.lname + ' = (blas_int) *' + arg.name + ';\n'
                    cleanup += tab + '*' + arg.name + ' = ' + arg.lname + ';\n'
            else:
                # output array
                query_args.append( arg.pname )
                proto_args.append( '\n    ' + arg.dtype + '* ' + arg.name )
                alias_args.append( arg.name )
                if (arg.is_array and (arg.dtype in ('int64_t', 'bool'))):
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
            if (arg.use_query and not arg.is_array):
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
        # aliases for real routines (hesv => sysv, etc.)
        s = re.search( r'^[sd](sy|sp|sb|or|op)(\w+)', func.xname )
        if (s):
            alias = alias_map[ s.group(1) ] + s.group(2)
            # todo: he
            txt += ('// ' + alias + ' alias to ' + func.name + '\n'
                +   'inline ' + func.retval + ' ' + alias + '(\n'
                +   tab + ', '.join( proto_args ) + ' )\n'
                +   '{\n'
                +   tab + 'return ' + func.name + '( ' + ', '.join( alias_args ) + ' );\n'
                +   '}\n\n')
        # end
    else:
        txt = (func.retval + ' ' + func.name + '(\n'
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
    txt = re.sub( r' +$', r'', txt, 0, re.M )
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

            if (arg.is_array):
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
                        s = re.search( r'^lapack::(\w+)', arg.dtype )
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
                if (arg.is_array):
                    lapacke_proto.append( 'lapack_int* ' + arg.name )
                else:
                    lapacke_proto.append( 'lapack_int ' + arg.name )
            else:
                if (arg.is_array or ('out' in arg.intent)):
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
        +  tab + 'params.time.value() = time;\n'
        +  tab + 'double gflop = lapack::Gflop< scalar_t >::' + func.name + '( ' + flop_args + ' );\n'
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
        +  tab*2 + 'params.ref_time.value() = time;\n'
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
    txt = re.sub( r' +$', r'', txt, 0, re.M )
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
    i = 0
    for f in files:
        i += 1
        last = (i == len(files))
        print( '    ' + f )
        func = parse_lapack( f )
        funcs.append( func )
        if (args.header):
            txt = generate_wrapper( func, header=True )
            print( txt, file=header, end='' )
        if (args.wrapper):
            txt = '// ' + '-'*77 + '\n'
            print( txt, file=wrapper, end='' )
            if (last):
                txt = generate_docs( func )
                print( txt, file=wrapper, end='' )
            else:
                print( '/// @ingroup ' + func.group + '\n', file=wrapper, end='' )
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
