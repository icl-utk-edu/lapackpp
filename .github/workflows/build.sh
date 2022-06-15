#!/bin/bash -e 

maker=$1
gpu=$2
echo Starting maker=$maker gpu=$gpu

source /etc/profile

export top=`pwd`

echo "======================================== load compiler"
date

module load gcc@7.3.0
module load intel-mkl

echo "======================================== verify dependencies"
# Check what is loaded.
module list

which g++
g++ --version

echo "MKLROOT ${MKLROOT}"

echo "======================================== env"
env

echo "======================================== setup build"
date
echo "maker ${maker}"
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

    (  # Buidl blaspp first
       git clone https://bitbucket.org/icl/blaspp
       mkdir blaspp/build && cd blaspp/build
       cmake -Dcolor=no -Dbuild_tests=no -DCMAKE_INSTALL_PREFIX=${top}/install ..
       make -j8 install
    )

    mkdir build && cd build
    cmake -Dcolor=no -DCMAKE_CXX_FLAGS="-Werror" \
          -DCMAKE_PREFIX_PATH=${top}/install/lib64/blaspp \
          -DCMAKE_INSTALL_PREFIX=${top}/install ..
fi

echo "======================================== build"
date
make -j8

echo "======================================== install"
date
make -j8 install
ls -R ${top}/install

echo "======================================== verify build"
date
ldd test/tester

echo "======================================== tests"
echo "Run tests."
date
cd test
export OMP_NUM_THREADS=8
./run_tests.py --quick --xml ${top}/report-${maker}.xml

echo "======================================== smoke tests"
echo "Verify install with smoke tests."
date
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

date
