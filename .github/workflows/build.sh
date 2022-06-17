#!/bin/bash -e 

maker=$1

mydir=`dirname $0`
source $mydir/setup_env.sh

section "======================================== build"
make -j8

section "======================================== install"
make -j8 install
ls -R ${top}/install

section "======================================== verify build"
ldd test/tester

