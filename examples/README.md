LAPACK++ Example
================================================================================

This is designed as a minimal, standalone example to demonstrate
how to include and link with LAPACK++. This assumes that LAPACK++ has
been compiled and installed. There are two options:

## Option 1: Makefile

The Makefile must know the compiler used to compile LAPACK++,
CXXFLAGS, and LIBS. Set CXX to the compiler, either in your environment
or in the Makefile. For the flags, there are two more options:

a. Using pkg-config to get CXXFLAGS and LIBS for LAPACK++ (recommended).
pkg-config must be able to locate the lapackpp package. If it is installed
outside the default search path (see `pkg-config --variable pc_path pkg-config`),
it should be added to `$PKG_CONFIG_PATH`. For instance, if it is installed
in /opt/slate:

    export PKG_CONFIG_PATH=/opt/slate/lib/pkgconfig  # for sh
    setenv PKG_CONFIG_PATH /opt/slate/lib/pkgconfig  # for csh

b. Hard-code CXXFLAGS and LIBS for LAPACK++ in the Makefile.

Then, to build and run `example_potrf` using the
Makefile, run:

    make
    make test

## Option 2: CMake

CMake must know the compiler used to compile LAPACK++. Set CXX to the
compiler, in your environment.

Create a build directory:

    mkdir build && cd build

If LAPACK++ is installed outside the default search path, tell cmake
where, for example, in /opt/slate:

    cmake -DCMAKE_PREFIX_PATH=/opt/slate ..

Otherwise, simply run:

    cmake ..

Then, to build and run `example_potrf` using the
resulting Makefile, run:

    make
    make test
