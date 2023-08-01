#!/bin/bash

#-------------------------------------------------------------------------------
# Functions

# Suppress echo (-x) output of commands executed with `quiet`.
# Useful for sourcing files, loading modules, spack, etc.
# set +x, set -x are not echo'd.
quiet() {
    { set +x; } 2> /dev/null;
    $@;
    set -x
}

# `print` is like `echo`, but suppresses output of the command itself.
# https://superuser.com/a/1141026
echo_and_restore() {
    builtin echo "$*"
    date
    case "${save_flags}" in
        (*x*)  set -x
    esac
}
alias print='{ save_flags="$-"; set +x; } 2> /dev/null; echo_and_restore'


#-------------------------------------------------------------------------------
quiet source /etc/profile

hostname && pwd
export top=$(pwd)

shopt -s expand_aliases


print "======================================== Load compiler"
quiet module load gcc@7.3.0
quiet which g++
g++ --version

quiet module load intel-oneapi-mkl
echo "MKLROOT=${MKLROOT}"

quiet module load pkgconf
quiet which pkg-config

# CMake finds CUDA in /usr/local/cuda, so need to explicitly set gpu_backend.
export gpu_backend=none
export color=no
export CXXFLAGS="-Werror -Wno-unused-command-line-argument"

# Test int64 build with make/cuda and cmake/amd.
# Test int32 build with cmake/cuda and make/amd and all others.
if [ "${maker}" = "make" -a "${device}" = "gpu_nvidia" ]; then
    export blas_int=int64
elif [ "${maker}" = "cmake" -a "${device}" = "gpu_amd" ]; then
    export blas_int=int64
else
    export blas_int=int32
fi

if [ "${device}" = "gpu_nvidia" ]; then
    print "======================================== Load CUDA"
    export CUDA_HOME=/usr/local/cuda/
    export PATH=${PATH}:${CUDA_HOME}/bin
    export gpu_backend=cuda
    quiet which nvcc
    nvcc --version
fi

if [ "${device}" = "gpu_amd" ]; then
    print "======================================== Load ROCm"
    export PATH=${PATH}:/opt/rocm/bin
    export gpu_backend=hip
    quiet which hipcc
    hipcc --version
fi

if [ "${device}" = "gpu_intel" ]; then
    print "======================================== Load Intel oneAPI"
    export gpu_backend=sycl
    quiet module load intel-oneapi-compilers
    quiet which icpx
    icpx --version
fi

if [ "${maker}" = "cmake" ]; then
    print "======================================== Load cmake"
    quiet module load cmake
    quiet which cmake
    cmake --version
    cd build
fi

quiet module list
