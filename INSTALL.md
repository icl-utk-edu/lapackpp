LAPACK++ Installation Notes
================================================================================

[TOC]

Synopsis
--------------------------------------------------------------------------------

Configure and compile the LAPACK++ library and its tester,
then install the headers and library.

Option 1: Makefile

    make && make install

Option 2: CMake

    mkdir build && cd build
    cmake ..
    make && make install

Makefile Installation
--------------------------------------------------------------------------------

    make           - configures (if make.inc is missing),
                     then compiles the library and tester.
    make config    - configures LAPACK++, creating a make.inc file.
    make lib       - compiles the library (lib/liblapackpp.so).
    make tester    - compiles test/tester.
    make docs      - generates documentation in docs/html/index.html
    make install   - installs the library and headers to ${prefix}.
    make uninstall - remove installed library and headers from ${prefix}.
    make clean     - deletes object (*.o) and library (*.a, *.so) files.
    make distclean - also deletes make.inc and dependency files (*.d).
    If static=1, makes .a instead of .so library.


### Details

    make config [options]

Runs the `configure.py` script to detect your compiler and library properties,
then creates a make.inc configuration file. You can also manually edit the
make.inc file. Options are name=value pairs to set variables. The configure.py
script can be invoked directly:

    python configure.py [options]

Running `configure.py -h` will print a help message with the current options.
Variables that affect configure.py include:

    CXX                C++ compiler
    CXXFLAGS           C++ compiler flags
    LDFLAGS            linker flags
    CPATH              compiler include search path
    LIBRARY_PATH       compile time library search path
    LD_LIBRARY_PATH    runtime library search path
    DYLD_LIBRARY_PATH  runtime library search path on macOS
    prefix             where to install:
                       headers go   in ${prefix}/include,
                       library goes in ${prefix}/lib${LIB_SUFFIX}

These can be set in your environment or on the command line, e.g.,

    python configure.py CXX=g++ prefix=/usr/local

Configure assumes environment variables are set so your compiler can find BLAS
and LAPACK libraries. For example:

    export LD_LIBRARY_PATH="/opt/my-blas/lib64"  # or DYLD_LIBRARY_PATH on macOS
    export LIBRARY_PATH="/opt/my-blas/lib64"
    export CPATH="/opt/my-blas/include"

or

    export LDFLAGS="-L/opt/my-blas/lib64 -Wl,-rpath,/opt/my-blas/lib64"
    export CXXFLAGS="-I/opt/my-blas/include"

On some systems, loading the appropriate module will set these flags:

    module load my-blas


### Vendor notes

Intel MKL provides scripts to set these flags, e.g.:

    source /opt/intel/bin/compilervars.sh intel64

or

    source /opt/intel/mkl/bin/mklvars.sh intel64

IBM ESSL provides only a subset of LAPACK functions,
so Netlib LAPACK is also required.


### Manual configuration

If you have a specific configuration that you want, set CXX, CXXFLAGS, LDFLAGS,
and LIBS, e.g.:

    export CXX="g++"
    export CXXFLAGS="-I${MKLROOT}/include -fopenmp"
    export LDFLAGS="-L${MKLROOT}/lib/intel64 -Wl,-rpath,${MKLROOT}/lib/intel64 -fopenmp"
    export LIBS="-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lm"

These can also be set when running configure:

    make config CXX=g++ \
                CXXFLAGS="-I${MKLROOT}/include -fopenmp" \
                LDFLAGS="-L${MKLROOT}/lib/intel64 -Wl,-rpath,${MKLROOT}/lib/intel64 -fopenmp" \
                LIBS="-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lm"

Note that all test programs are compiled with those options, so errors may cause
configure to fail.

If you experience unexpected problems, please see config/log.txt to diagnose the
issue. The log shows the option being tested, the exact command run, the
command's standard output (stdout), error output (stderr), and exit status. All
test files are in the config directory.

CMake Installation
--------------------------------------------------------------------------------

Note that LAPACK++ inherits its dependencies from BLAS++. It requires th
BLAS++ library to be installed via CMake prior to compilation.  Information and
installation instructions can be found at https://bitbucket.org/icl/blaspp.

The CMake script enforces an out of source build. The simplest way to accomplish
this is to create a build directory off the LAPACK++ root directory:

    cd /my/lapackpp/dir
    mkdir build && cd build

### Options

By default LAPACK++ is set to install into `/opt/slate/`. If you wish to
change this, CMake needs to be told where to install the LAPACK++ library.
You can do this by defining CMAKE_INSTALL_PREFIX variable via the CMake
command line:

    # Assuming the working dir is still /my/lapackpp/dir/build
    cmake -DCMAKE_INSTALL_PREFIX=/path/to/my/dir ..

By default LAPACK++ builds a testing suite located in `lapackpp/test`.  To disable,
define `LAPACKPP_BUILD_TESTS` as `OFF`, as follows:

    # Disable building LAPACKPP test suite
    cmake -DLAPACKPP_BUILD_TESTS=OFF ..

If `LAPACKPP_BUILD_TESTS` is enabled, the build will require the TestSweeper
library to be installed via CMake prior to compilation.  Information and
installation instructions can be found at https://bitbucket.org/icl/testsweeper.

### LAPACK++ Library options

LAPACK++ inherits its dependencies from BLAS++ as noted above.  However, if the
user wishes to override these options, they may set `USE_OPTIMIZED_LAPACK` to `TRUE`
to use the CMake included `find_package(LAPACK)`.

The user may also set `LAPACK_LIBRARIES` to the path of their desired LAPACK
library.  LAPACK++ will then attempt to explicitly link to this.

Once the LAPACK library is set or inherited, the CMake script attempts to compile
several small code snippets to determine what compiler options are necessary for
LAPACK++.

### CMake build
Once CMake generates the required makefiles, BLAS++ can be built
and installed using the following:

    # Assuming the working dir is still /my/lapackpp/dir/build
    make
    make install
