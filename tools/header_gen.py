#!/usr/bin/env python
#
# generate C prototypes for the LAPACK Fortran functions by parsing
# doxygen comments in LAPACK's SRC *.f files.

from __future__ import print_function

import sys
import re
import os

lapack = os.environ['LAPACKDIR']
if (len(sys.argv) == 1):
    funcs = []
    #f = open( lapack + '/LAPACKE/include/lapack.h' )
    f = open( '../include/lapack_fortran.h' )
    for line in f:
        s = re.search( r' +LAPACK_(\w+) +LAPACK_GLOBAL\( *\1', line )
        #s = re.search( r'(?:void|float|double) +LAPACK_GLOBAL\((\w+),', line )
        if (s):
            funcs.append( s.group(1) )
    # end
    output = open( 'lapack_parse.h', 'w' )
else:
    funcs = sys.argv[1:]
    output = sys.stdout
# end

intent = None
var    = None
vtype  = None
dtype  = None

typemap = {
    'character'        : 'char',
    'integer'          : 'lapack_int',
    'real'             : 'float',
    'double precision' : 'double',
    'complex'          : 'lapack_complex_float',
    'complex*16'       : 'lapack_complex_double',
    'logical'          : 'lapack_logical',

    'logical function of two real arguments'               : 'LAPACK_S_SELECT2',
    'logical function of three real arguments'             : 'LAPACK_S_SELECT3',

    'logical function of two double precision arguments'   : 'LAPACK_D_SELECT2',
    'logical function of three double precision arguments' : 'LAPACK_D_SELECT3',

    'logical function of one complex argument'     : 'LAPACK_C_SELECT1',
    'logical function of two complex arguments'    : 'LAPACK_C_SELECT2',

    'logical function of one complex*16 argument'  : 'LAPACK_Z_SELECT1',
    'logical function of two complex*16 arguments' : 'LAPACK_Z_SELECT2',
}

for func in funcs:
    s = re.search( '\.f', func )
    if (s):
        filename = func
    else:
        for subdir in ('SRC', 'INSTALL', 'TESTING/MATGEN'):
            filename = lapack + '/' + subdir + '/' + func + '.f'
            if (os.path.exists( filename )):
                break
            # end
        # end
    # end

    if (os.path.exists( filename )):
        #print( 'read', filename )
        f = open( filename )
        args = []
        retval = 'void'
        for line in f:
            s = re.search( r'^\* +\b(.*)\b +FUNCTION +(\w+)', line )  # \( *(.*) *\)
            if (s):
                retval = typemap[ s.group(1).lower() ]
                continue
            # end

            s = re.search( r'^\*> *\\param\[(in|out|in,out|inout)\] +(\w+)', line )
            if (s):
                intent = s.group(1).lower()
                var    = s.group(2)
                vtype  = var  # vtype used as flag to say we have a variable and are looking for its type
                continue
            # end

            if (vtype):
                s = (re.search( '^\*> *' + vtype + ' +is +(CHARACTER|INTEGER|REAL|DOUBLE PRECISION|COMPLEX\*16|COMPLEX|LOGICAL)( array)?', line ) or
                     re.search( '^\*> *\(workspace\) +(CHARACTER|INTEGER|REAL|DOUBLE PRECISION|COMPLEX\*16|COMPLEX|LOGICAL)( array)?', line ) or
                     re.search( '^\*> *' + vtype + ' +is a (LOGICAL FUNCTION of (?:one|two|three) (?:REAL|DOUBLE PRECISION|COMPLEX\*16|COMPLEX) arguments?)( array)?', line ))
                if (s):
                    dtype = s.group(1).lower()
                    dtype = typemap[dtype]
                    array = s.group(2)
                    vtype = None
                    # leave arrays uppercase, except work and pivots
                    if (array and var not in ('WORK', 'RWORK', 'IWORK', 'IPIV', 'JPIV', 'PIV')):
                        pass #lvar = var
                    else:
                        var = var.lower()
                    if (intent == 'in'):
                        args.append( dtype + ' const* ' + var )
                    else:
                        args.append( dtype + '* ' + var )
                    continue
                # end
            # end
        # end
        print( '#define LAPACK_' + func + ' LAPACK_GLOBAL(' + func + ',' + func.upper() + ')', file=output )
        print( retval + ' LAPACK_' + func + '( ' + ', '.join( args ) + ' );', file=output )
    else:
        print( 'skipping, file not found:', filename )
        print( '#define LAPACK_' + func + ' LAPACK_GLOBAL(' + func + ',' + func.upper() + ')', file=output )
        print( 'void LAPACK_' + func + '( ... );', file=output )
    # end
# end

output.close()
