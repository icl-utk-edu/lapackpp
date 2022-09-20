#!/bin/bash -e

maker=$1

if [ "${maker}" = "cmake" ]; then
    rm -rf build
    mkdir -p build
fi

mydir=`dirname $0`
source $mydir/setup_env.sh

section "======================================== setup build"

export color=no
rm -rf ${top}/install
if [ "${maker}" = "make" ]; then
    make distclean
    make config CXXFLAGS="-Werror" prefix=${top}/install
fi
if [ "${maker}" = "cmake" ]; then
    module load cmake
    which cmake
    cmake --version

    (  # Build blaspp first
       git clone https://github.com/icl-utk-edu/blaspp
       mkdir blaspp/build && cd blaspp/build
       cmake -Dcolor=no -Dbuild_tests=no -DCMAKE_INSTALL_PREFIX=${top}/install ..
       make -j8 install
    )

    cmake -Dcolor=no -DCMAKE_CXX_FLAGS="-Werror" \
          -DCMAKE_PREFIX_PATH=${top}/install/lib64/blaspp \
          -DCMAKE_INSTALL_PREFIX=${top}/install ..
fi

