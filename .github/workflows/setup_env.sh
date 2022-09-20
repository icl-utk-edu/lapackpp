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

# `section` is like `echo`, but suppresses output of the command itself.
# https://superuser.com/a/1141026
print_section() {
    builtin echo "$*"
    date
    case "$save_flags" in
        (*x*)  set -x
    esac
}
alias section='{ save_flags="$-"; set +x; } 2> /dev/null; print_section'


#-------------------------------------------------------------------------------
quiet source /etc/profile

hostname && pwd
export top=`pwd`

shopt -s expand_aliases


section "======================================== Load compiler"
quiet module load gcc@7.3.0
quiet module load intel-mkl

if [ "${maker}" = "cmake" ]; then
    section "======================================== Load cmake"
    quiet module load cmake
    which cmake
    cmake --version
    cd build
fi
