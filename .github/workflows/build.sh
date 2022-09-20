#!/bin/bash -xe

maker=$1

mydir=`dirname $0`
source $mydir/setup_env.sh

section "======================================== Build"
make -j8

section "======================================== Install"
make -j8 install
ls -R ${top}/install

section "======================================== Verify build"
ldd test/tester

