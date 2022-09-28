#!/usr/bin/env python3

'''
Generate C prototypes for the LAPACK Fortran functions by parsing
doxygen comments in LAPACK's SRC *.f files. Looks for routines in
LAPACK's SRC, INSTALL, and MATGEN directories. (Because MATGEN is the
last directory checked, error messages show that file not found.)

Example usage:

    # Assumes $LAPACKDIR is set
    lapackpp> echo $LAPACKDIR
    /Users/mgates/Documents/lapack

    lapackpp> ./tools/header_gen.py {s,d,c,z}posv {s,d}ormqr {c,z}unmqr
    generating gen/fortran.h

    lapackpp> ./tools/header_gen.py all
    reading include/lapack/fortran.h to re-generate all prototypes
    generating gen/fortran.h
    // skipping, file not found: /Users/mgates/Documents/lapack/TESTING/MATGEN/slassq.f
    // skipping, file not found: /Users/mgates/Documents/lapack/TESTING/MATGEN/dlassq.f
    // skipping, file not found: /Users/mgates/Documents/lapack/TESTING/MATGEN/classq.f
    // skipping, file not found: /Users/mgates/Documents/lapack/TESTING/MATGEN/zlassq.f
    // skipping, file not found: /Users/mgates/Documents/lapack/TESTING/MATGEN/slartg.f
    // skipping, file not found: /Users/mgates/Documents/lapack/TESTING/MATGEN/dlartg.f
    // skipping, file not found: /Users/mgates/Documents/lapack/TESTING/MATGEN/clartg.f
    // skipping, file not found: /Users/mgates/Documents/lapack/TESTING/MATGEN/zlartg.f
'''

from __future__ import print_function

import sys
import re
import os

lapack = os.environ['LAPACKDIR']

if (len( sys.argv ) == 1):
    print( 'Usage:', sys.argv[0], '[routine ...]\n' +
           '      ', sys.argv[0], 'all\n' +
           'See script for more details' )
    exit(1)
# end

if (sys.argv[1] == 'all'):
    #header = lapack + '/LAPACKE/include/lapack.h'
    header = 'include/lapack/fortran.h'
    print( 'reading', header, 'to re-generate all prototypes' )
    funcs = []
    f = open( header )
    for line in f:
        s = re.search( r' +LAPACK_(\w+) +LAPACK_GLOBAL\( *\1', line )
        #s = re.search( r'(?:void|float|double) +LAPACK_GLOBAL\((\w+),', line )
        if (s):
            funcs.append( s.group(1) )
    # end
else:
    funcs = sys.argv[1:]
# end

output = 'gen/fortran.h'
print( 'generating', output )
output = open( output, 'w' )

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
        print( retval + ' LAPACK_' + func + '(\n    ' + ', '.join( args ) + '\n);\n', file=output )
    else:
        print( '// skipping, file not found:', filename )
        print( '// skipping, file not found:', filename, file=output )
        print( '// #define LAPACK_' + func + ' LAPACK_GLOBAL(' + func + ',' + func.upper() + ')', file=output )
        print( '// void LAPACK_' + func + '( ... );\n', file=output )
    # end
# end

output.close()
