#!/bin/bash -e 

maker=$1

source .github/workflows/setup_env.sh

section "======================================== build"
make -j8

section "======================================== install"
make -j8 install
ls -R ${top}/install

section "======================================== verify build"
ldd test/tester

