#!/bin/bash

shopt -s expand_aliases

set +x

source /etc/profile

top=`pwd`

# Suppress echo (-x) output of commands executed with `quiet`.
quiet() {
    { set +x; } 2> /dev/null;
    $@;
    set -x
}

print_section() {
    builtin echo "$*"
    date
    case "$save_flags" in
        (*x*)  set -x
    esac
}
alias section='{ save_flags="$-"; set +x; } 2> /dev/null; print_section'

module load gcc@7.3.0
module load intel-mkl

if [ "${maker}" = "cmake" ]; then
    section "======================================== Load cmake"
    module load cmake
    which cmake
    cmake --version
    rm -rf build && mkdir -p build && cd build
fi


section "======================================== Verify dependencies"
module list

which g++
g++ --version

echo "MKLROOT=${MKLROOT}"

section "======================================== Environment"
env

set -x

