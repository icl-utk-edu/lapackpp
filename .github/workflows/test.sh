#!/bin/bash -xe

maker=$1
device=$2

mydir=$(dirname $0)
source ${mydir}/setup_env.sh

section "======================================== Tests"
cd test
export OMP_NUM_THREADS=8
./run_tests.py --host --quick --xml ${top}/report-${maker}.xml

# CUDA or HIP
./run_tests.py --device --quick --xml ${top}/report-${maker}-device.xml


section "======================================== Smoke tests"
cd ${top}/example

if [ "${maker}" = "make" ]; then
    export PKG_CONFIG_PATH=${top}/install/lib/pkgconfig
    make clean
fi
if [ "${maker}" = "cmake" ]; then
    rm -rf build && mkdir build && cd build
    cmake "-DCMAKE_PREFIX_PATH=${top}/install/lib64/blaspp;${top}/install/lib64/lapackpp" ..
fi

make
./example_potrf || exit 1

section "======================================== Finished"
