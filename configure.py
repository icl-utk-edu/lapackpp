#!/usr/bin/python
#
# Usage: python configure.py [--interactive]
#
# Note: using #!/usr/bin/env python or #!/usr/bin/python doesn't work,
# because LD_LIBRARY_PATH isn't propogated.

from __future__ import print_function

import sys
import re
import config
from   config import ansi_bold, ansi_red, ansi_blue, ansi_normal
from   config import Error
import config.lapack

#-------------------------------------------------------------------------------
# main

print( '-'*80 + '\n' +
ansi_bold + ansi_red + '                              Welcome to LAPACK++.' +
ansi_normal + '''

By default, this configuration script will automatically choose the first valid
value it finds for each option. You can set it to interactive to find all
possible values and give you a choice:
    ''' + ansi_blue + 'python configure.py --interactive' + ansi_normal + '''

To select versions of BLAS and LAPACK to check for, set one or more of:
    mkl=1  acml=1  essl=1  openblas=1  accelerate=1
For instance,
    make config CXX=xlc++ essl=1

This script assumes flags are set in your environment to make your BLAS
and LAPACK libraries accessible to your compiler, for example:
    export LD_LIBRARY_PATH="/opt/my-blas/lib64"  # or DYLD_LIBRARY_PATH on MacOS
    export LIBRARY_PATH="/opt/my-blas/lib64"
    export CPATH="/opt/my-blas/include"
or
    export LDFLAGS="-L/opt/my-blas/lib64 -Wl,-rpath,/opt/my-blas/lib64"
    export CXXFLAGS="-I/opt/my-blas/include"
On some systems, loading the appropriate module will set these flags:
    module load my-blas
Intel MKL provides a script to set these flags:
    source /opt/intel/bin/compilervars.sh intel64
or
    source /opt/intel/mkl/bin/mklvars.sh intel64
If you have a specific configuration that you want, set CXX, CXXFLAGS, LDFLAGS,
and LIBS, e.g.:
    export CXX="g++"
    export CXXFLAGS="-I${MKLROOT}/include -fopenmp"
    export LDFLAGS="-L${MKLROOT}/lib/intel64"
    export LIBS="-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lm"
''' + '-'*80 )

config.init( prefix='/usr/local/lapackpp' )
config.prog_cxx()
config.prog_cxx_flags([
    '-O2', '-std=c++11', '-MMD',
    '-Wall',
    '-pedantic',
    '-Wshadow',
    '-Wno-unused-local-typedefs',
    '-Wno-unused-function',
    #'-Wmissing-declarations',
    #'-Wconversion',
    #'-Werror',
])

config.lapack.blas()
print()
config.lapack.blas_float_return()
config.lapack.blas_complex_return()

# If we can, be smart looking for MKL, ESSL, or OpenBLAS version;
# otherwise, check them all.
LIBS = config.environ['LIBS']
if (re.search( 'mkl', LIBS )):
    config.lapack.mkl_version()
elif (re.search( 'essl', LIBS )):
    config.lapack.essl_version()
elif (re.search( 'openblas', LIBS )):
    config.lapack.openblas_version()
else:
    config.lapack.mkl_version()
    config.lapack.essl_version()
    config.lapack.openblas_version()

# Must test mkl_version before cblas and lapacke, to define HAVE_MKL.
try:
    config.lapack.cblas()
except Error:
    print( ansi_red + 'LAPACK++ needs CBLAS only in test_syr.' + ansi_normal )
    pass

config.lapack.lapack()
config.lapack.lapack2()
config.lapack.lapack_version()
config.lapack.lapack_xblas()
config.lapack.lapack_matgen()

try:
    config.lapack.lapacke()
    config.lapack.lapacke2()
except Error:
    print( ansi_red + 'LAPACK++ needs LAPACKE only in testers.' + ansi_normal )
    pass

config.output_files( 'make.inc' )

print( '-'*80 )
