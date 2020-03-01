#!/usr/bin/env python
#
# Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
#
# Usage: python configure.py [--interactive]

from __future__ import print_function

import sys
import re
import config
from   config import Error, font, print_warn
import config.lapack

#-------------------------------------------------------------------------------
# header

print( '-'*80 + '\n' +
font.bold( font.blue( '                              Welcome to LAPACK++.' ) ) +
'''

By default, configure will automatically choose the first valid value it finds
for each option. You can set it to interactive to find all possible values and
give you a choice:
    ''' + font.blue( 'make config interactive=1' ) + '''

If you have multiple compilers, we suggest specifying your desired compiler by
setting CXX, as the automated search may prefer a different compiler.

To limit which versions of BLAS and LAPACK to search for, set one of:
    blas=mkl
    blas=acml
    blas=essl
    blas=openblas
    blas=accelerate
For instance,
    ''' + font.blue( 'make config CXX=xlc++ blas=essl' ) + '''

Some BLAS libraries have 32-bit int (lp64) or 64-bit int (ilp64) variants.
Configure will auto-detect a scheme, but you can also specify it by setting:
    lp64=1
    ilp64=1

BLAS and LAPACK are written in Fortran, which has a compiler-specific name
mangling scheme: routine DGEMM is called dgemm_, dgemm, or DGEMM in the
library. (Some libraries like MKL and ESSL support multiple schemes.)
Configure will auto-detect a scheme, but you can also specify it by setting:
    fortran_mangling=add_
    fortran_mangling=lower
    fortran_mangling=upper

For ANSI colors, set color=auto (when output is TTY), color=yes, or color=no.

Configure assumes environment variables CPATH, LIBRARY_PATH, and LD_LIBRARY_PATH
are set so your compiler can find libraries. See INSTALL.txt for more details.
''' + '-'*80 )

#-------------------------------------------------------------------------------
def main():
    config.init( prefix='/usr/local/lapackpp' )
    config.prog_cxx()
    config.prog_cxx_flags([
        '-O2', '-std=c++11', '-MMD',
        '-Wall',
        '-pedantic',
        '-Wshadow',
        '-Wno-unused-local-typedefs',
        '-Wno-unused-function',
        #'-Wmissing-declarations',
        #'-Wconversion',
        #'-Werror',
    ])
    config.openmp()

    config.lapack.blas()
    print()
    config.lapack.blas_float_return()
    config.lapack.blas_complex_return()
    config.lapack.vendor_version()

    # Must test mkl_version before cblas and lapacke, to define HAVE_MKL.
    try:
        config.lapack.cblas()
    except Error:
        print_warn( 'LAPACK++ needs CBLAS only in testers.' )

    # todo: can LAPACK++ be compiled without uncommon routines?
    config.lapack.lapack()
    config.lapack.lapack_uncommon()
    config.lapack.lapack_version()

    # XBLAS and Matgen are optional
    try:
        config.lapack.lapack_xblas()
    except Error:
        print_warn( 'LAPACK++ will exclude wrappers for XBLAS.' )

    try:
        config.lapack.lapack_matgen()
    except Error:
        print_warn( 'LAPACK++ will exclude wrappers for matgen.' )

    try:
        config.lapack.lapacke()
        config.lapack.lapacke_uncommon()
    except Error:
        print_warn( 'LAPACK++ needs LAPACKE only in testers.' )

    blaspp = config.get_package(
        'BLAS++',
        ['../blaspp', './blaspp'],
        'https://bitbucket.org/icl/blaspp',
        'https://bitbucket.org/icl/blaspp/get/tip.tar.gz',
        'blaspp.tar.gz' )
    if (not blaspp):
        raise Exception( 'LAPACK++ requires BLAS++.' )

    testsweeper = config.get_package(
        'testsweeper',
        ['../testsweeper', blaspp + '/testsweeper', './testsweeper'],
        'https://bitbucket.org/icl/testsweeper',
        'https://bitbucket.org/icl/testsweeper/get/tip.tar.gz',
        'testsweeper.tar.gz' )
    if (not testsweeper):
        print_warn( 'LAPACK++ needs TestSweeper only in testers.' )

    config.extract_defines_from_flags( 'CXXFLAGS' )
    config.output_files( ['make.inc', 'lapack_defines.h'] )
    print( 'log in config/log.txt' )

    print( '-'*80 )
# end

#-------------------------------------------------------------------------------
try:
    main()
except Error as ex:
    print_warn( 'A fatal error occurred. ' + str(ex) +
                '\nBLAS++ could not be configured. Log in config/log.txt' )
    exit(1)
