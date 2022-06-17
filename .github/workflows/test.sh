#!/bin/bash -e 

maker=$1

source .github/workflows/setup_env.sh

section "======================================== Run tests"
cd test
export OMP_NUM_THREADS=8
./run_tests.py --quick --xml ${top}/report-${maker}.xml

section "======================================== Smoke tests"
cd ${top}/example

if [ "${maker}" = "make" ]; then
    export PKG_CONFIG_PATH=${top}/install/lib/pkgconfig
    make clean
fi
if [ "${maker}" = "cmake" ]; then
    mkdir build && cd build
    cmake "-DCMAKE_PREFIX_PATH=${top}/install/lib64/blaspp;${top}/install/lib64/lapackpp" ..
fi

make
./example_potrf || exit 1

section "======================================== Finished"
