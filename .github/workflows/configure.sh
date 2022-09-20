#!/bin/bash -xe

maker=$1

if [ "${maker}" = "cmake" ]; then
    rm -rf build
    mkdir -p build
fi

mydir=`dirname $0`
source $mydir/setup_env.sh

section "======================================== Verify dependencies"
quiet module list
quiet which g++
quiet g++ --version

echo "MKLROOT=${MKLROOT}"

section "======================================== Environment"
env

section "======================================== Setup build"
export color=no
rm -rf ${top}/install
if [ "${maker}" = "make" ]; then
    make distclean
    make config CXXFLAGS="-Werror" prefix=${top}/install
fi
if [ "${maker}" = "cmake" ]; then
    (  # Build blaspp first
       git clone https://github.com/icl-utk-edu/blaspp
       mkdir blaspp/build && cd blaspp/build
       cmake -Dcolor=no -Dbuild_tests=no -DCMAKE_INSTALL_PREFIX=${top}/install ..
       make -j8 install
    )

    cmake -Dcolor=no -DCMAKE_CXX_FLAGS="-Werror" \
          -DCMAKE_INSTALL_PREFIX=${top}/install ..
fi

