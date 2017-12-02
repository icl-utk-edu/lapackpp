#!/usr/bin/env python
#
# Example usage:
# help
#     ./run_tests.py -h
#
# run everything with default sizes
# output is redirected; summary information is printed on stderr
#     ./run_tests.py > output.txt
#
# run LU (gesv, getrf, getri, ...), Cholesky (posv, potrf, potri, ...)
# with float, double and default sizes
#     ./run_tests.py -f -d --lu --chol
#
# run getrf, potrf with small, medium sizes
#     ./run_tests.py -s -m getrf potrf

from __future__ import print_function

import sys
import os
import re
import argparse

# ------------------------------------------------------------------------------
# command line arguments
parser = argparse.ArgumentParser()

group1 = parser.add_argument_group( 'precisions (default is all)' )
group1.add_argument( '-f', '--float',          action='store_true', help='run float (single precision) tests' )
group1.add_argument( '-d', '--double',         action='store_true', help='run double precision tests' )
group1.add_argument( '-c', '--complex-float',  action='store_true', help='run complex-float precision tests' )
group1.add_argument( '-z', '--complex-double', action='store_true', help='run complex-double precision tests' )

group2 = parser.add_argument_group( 'matrix dimensions (default is medium)' )
group2.add_argument( '-s', '--small',  action='store_true', help='run small tests' )
group2.add_argument( '-m', '--medium', action='store_true', help='run medium tests' )
group2.add_argument( '-l', '--large',  action='store_true', help='run large tests' )

group3 = parser.add_argument_group( 'category (default is all)' )
categories = [
    group3.add_argument( '--lu',            action='store_true', help='run LU tests' ),
    group3.add_argument( '--chol',          action='store_true', help='run Cholesky tests' ),
    group3.add_argument( '--sysv',          action='store_true', help='run symmetric indefinite (Bunch-Kaufman) tests' ),
    group3.add_argument( '--rook',          action='store_true', help='run symmetric indefinite (rook) tests' ),
    group3.add_argument( '--aasen',         action='store_true', help='run symmetric indefinite (Aasen) tests' ),
    group3.add_argument( '--least-squares', action='store_true', help='run least squares tests' ),
    group3.add_argument( '--qr',            action='store_true', help='run QR tests' ),
    group3.add_argument( '--lq',            action='store_true', help='run LQ tests' ),
    group3.add_argument( '--ql',            action='store_true', help='run QL tests' ),
    group3.add_argument( '--rq',            action='store_true', help='run RQ tests' ),
    group3.add_argument( '--syev',          action='store_true', help='run symmetric eigenvalues tests' ),
    group3.add_argument( '--geev',          action='store_true', help='run non-symmetric eigenvalues tests' ),
    group3.add_argument( '--svd',           action='store_true', help='run svd tests' ),
    group3.add_argument( '--aux',           action='store_true', help='run auxiliary tests' ),
    group3.add_argument( '--aux-house',     action='store_true', help='run auxiliary Householder tests' ),
    group3.add_argument( '--aux-norm',      action='store_true', help='run auxiliary norm tests' ),
    group3.add_argument( '--blas',          action='store_true', help='run additional BLAS tests' ),
]
categories = map( lambda x: x.dest, categories ) # map to names: ['lu', 'chol', ...]

parser.add_argument( 'tests', nargs=argparse.REMAINDER )
args = parser.parse_args()

# by default, run all precisions
if (not (args.float or args.double or args.complex_float or args.complex_double)):
    args.float          = True
    args.double         = True
    args.complex_float  = True
    args.complex_double = True

# by default, run medium sizes
if (not (args.small or args.medium or args.large)):
    args.medium = True

# by default, run all categories
if (args.tests or not any( map( lambda c: args.__dict__[ c ], categories ))):
    for c in categories:
        args.__dict__[ c ] = True

# ------------------------------------------------------------------------------
# parameters
# begin with space to ease concatenation

square   = ''
tall     = ''
wide     = ''
mnk      = ''
nk_tall  = ''
nk_wide  = ''

if (args.small):
    square  += ' --dim 25:100:25'
    tall    += ' --dim 50:200:50x25:100:25'  # 2:1
    wide    += ' --dim 25:100:25x50:200:50'  # 1:2
    mnk     += ' --dim 25x50x75 --dim 50x25x75' \
            +  ' --dim 25x75x50 --dim 50x75x25' \
            +  ' --dim 75x25x50 --dim 75x50x25'
    nk_tall += ' --dim 1x50:200:50x25:100:25'
    nk_wide += ' --dim 1x25:100:25x50:200:50'

if (args.medium):
    square  += ' --dim 100:500:100'
    tall    += ' --dim 200:1000:200x100:500:100'  # 2:1
    wide    += ' --dim 100:500:100x200:1000:200'  # 1:2
    mnk     += ' --dim 100x300x600 --dim 300x100x600' \
            +  ' --dim 100x600x300 --dim 300x600x100' \
            +  ' --dim 600x100x300 --dim 600x300x100'
    nk_tall += ' --dim 1x200:1000:200x100:500:100'
    nk_wide += ' --dim 1x100:500:100x200:1000:200'

if (args.large):
    square  += ' --dim 1000:5000:1000'
    tall    += ' --dim 2000:10000:2000x1000:5000:1000'  # 2:1
    wide    += ' --dim 1000:5000:1000x2000:10000:2000'  # 1:2
    mnk     += ' --dim 1000x3000x6000 --dim 3000x1000x6000' \
            +  ' --dim 1000x6000x3000 --dim 3000x6000x1000' \
            +  ' --dim 6000x1000x3000 --dim 6000x3000x1000'
    nk_tall += ' --dim 1x2000:10000:2000x1000:5000:1000'
    nk_wide += ' --dim 1x1000:5000:1000x2000:10000:2000'

mn       = square + tall + wide
mnk      = mn + mnk
nk       = square + nk_tall + nk_wide

incx_pos = ' --incx 1,2'
incx     = ' --incx 1,2,-1,-2'
incy_pos = ' --incy 1,2'
incy     = ' --incy 1,2,-1,-2'

dtypes = []
if (args.float):  dtypes.append( 's' )
if (args.double): dtypes.append( 'd' )
if (args.complex_float):  dtypes.append( 'c' )
if (args.complex_double): dtypes.append( 'z' )
dtype         = ' --type=' + ','.join( dtypes )

r = filter( lambda x: x in ('s', 'd'), dtypes )
if (r):
    dtype_real = ' --type=' + ','.join( r )
else:
    dtype_real = ''

c = filter( lambda x: x in ('c', 'z'), dtypes )
if (c):
    dtype_complex = ' --type=' + ','.join( c )
else:
    dtype_complex = ''

trans    = ' --trans n,t,c'
trans_nt = ' --trans n,t'
trans_nc = ' --trans n,c'
uplo     = ' --uplo l,u'
norm     = ' --norm 1,inf,fro,max'
diag     = ' --diag n,u'
direct   = ' --direct f,b'
storev   = ' --storev c,r'
side     = ' --side l,r'
mtype    = ' --matrixtype g,l,u'

# ------------------------------------------------------------------------------
cmds = []

# LU
if (args.lu):
    cmds += [
    [ 'gesv',  dtype + square ],
    [ 'getrf', dtype + mn ],
    [ 'getrs', dtype + square + trans ],
    [ 'getri', dtype + square ],
    [ 'gecon', dtype + square ],
    [ 'gerfs', dtype + square + trans ],
    [ 'geequ', dtype + square ],
    ]

# Cholesky
if (args.chol):
    cmds += [
    [ 'posv',  dtype + square + uplo ],
    [ 'potrf', dtype + square + uplo ],
    [ 'potrs', dtype + square + uplo ],
    [ 'potri', dtype + square + uplo ],
    [ 'pocon', dtype + square + uplo ],
    [ 'porfs', dtype + square + uplo ],
    [ 'poequ', dtype + square ],  # only diagonal elements
    ]

# symmetric indefinite, Bunch-Kaufman
if (args.sysv):
    cmds += [
    [ 'sysv',  dtype + square + uplo ],
    [ 'sytrf', dtype + square + uplo ],
    [ 'sytrs', dtype + square + uplo ],
    [ 'sytri', dtype + square + uplo ],
    [ 'sycon', dtype + square + uplo ],
    [ 'syrfs', dtype + square + uplo ],
    ]

# symmetric indefinite, rook
#if (args.rook):
#    cmds += [
#    [ 'sysv_rook',  dtype + square + uplo ],
#    [ 'sytrf_rook', dtype + square + uplo ],
#    [ 'sytrs_rook', dtype + square + uplo ],
#    [ 'sytri_rook', dtype + square + uplo ],
#    ]

# symmetric indefinite, Aasen
#if (args.aasen):
#    cmds += [
#    [ 'sysv_aasen',  dtype + square + uplo ],
#    [ 'sytrf_aasen', dtype + square + uplo ],
#    [ 'sytrs_aasen', dtype + square + uplo ],
#    [ 'sytri_aasen', dtype + square + uplo ],
#    [ 'sysv_aasen_2stage',  dtype + square + uplo ],
#    [ 'sytrf_aasen_2stage', dtype + square + uplo ],
#    [ 'sytrs_aasen_2stage', dtype + square + uplo ],
#    [ 'sytri_aasen_2stage', dtype + square + uplo ],
#    ]

# least squares
#if (args.least_squares):
#    cmds += [
#    [ 'gels',   dtype + mn ],
#    [ 'gelsy',  dtype + mn ],
#    [ 'gelsd',  dtype + mn ],
#    [ 'gelss',  dtype + mn ],
#    [ 'getsls', dtype + mn ],
#    ]

# QR
#if (args.qr):
#    cmds += [
#    [ 'geqrf', dtype + mn ],
#    [ 'ggqrf', dtype + mn ],
#    [ 'ungqr', dtype + mn ],
#    [ 'unmqr', dtype + mn ],
#    ]

# LQ
#if (args.lq):
#    cmds += [
#    [ 'gelqf', dtype + mn ],
#    [ 'gglqf', dtype + mn ],
#    [ 'unglq', dtype + mn ],
#    [ 'unmlq', dtype + mn ],
#    ]

# QL
#if (args.ql):
#    cmds += [
#    [ 'geqlf', dtype + mn ],
#    [ 'ggqlf', dtype + mn ],
#    [ 'ungql', dtype + mn ],
#    [ 'unmql', dtype + mn ],
#    ]

# RQ
#if (args.rq):
#    cmds += [
#    [ 'gerqf', dtype + mn ],
#    [ 'ggrqf', dtype + mn ],
#    [ 'ungrq', dtype + mn ],
#    [ 'unmrq', dtype + mn ],
#    ]

# symmetric eigenvalues
# todo: add jobs
#if (args.syev):
#    cmds += [
#    [ 'syev',  dtype + square + uplo ],
#    [ 'syevx', dtype + square + uplo ],
#    [ 'syevd', dtype + square + uplo ],
#    [ 'syevr', dtype + square + uplo ],
#    [ 'sytrd', dtype + square + uplo ],
#    [ 'orgtr', dtype + square + uplo ],
#    [ 'ormtr', dtype + square + uplo ],
#    ]

# generalized symmetric eigenvalues
# todo: add jobs
#if (args.sygv):
#    cmds += [
#    [ 'sygv',  dtype + square + uplo ],
#    [ 'sygvx', dtype + square + uplo ],
#    [ 'sygvd', dtype + square + uplo ],
#    [ 'sygvr', dtype + square + uplo ],
#    [ 'sygst', dtype + square + uplo ],
#    ]

# non-symmetric eigenvalues
# todo: add jobs
#if (args.syev):
#    cmds += [
#    [ 'syev',  dtype + mn ],
#    [ 'syevx', dtype + mn ],
#    [ 'syevd', dtype + mn ],
#    [ 'syevr', dtype + mn ],
#    [ 'sytrd', dtype + mn ],
#    [ 'orgtr', dtype + mn ],
#    [ 'ormtr', dtype + mn ],
#    ]

# svd
# todo: add jobs
#if (args.svd):
#    cmds += [
#    [ 'gesvd',         dtype + mn ],
#    [ 'gesdd',         dtype + mn ],
#    [ 'gesvdx',        dtype + mn ],
#    [ 'gesvd_2stage',  dtype + mn ],
#    [ 'gesdd_2stage',  dtype + mn ],
#    [ 'gesvdx_2stage', dtype + mn ],
#    [ 'gejsv',         dtype + mn ],
#    [ 'gesvj',         dtype + mn ],
#    ]

# auxilary
if (args.aux):
    cmds += [
    [ 'lacpy', dtype + mn + mtype ],
    [ 'laset', dtype + mn + mtype ],
    [ 'laswp', dtype + mn ],
    ]

# auxilary - householder
if (args.aux_house):
    cmds += [
    [ 'larfg', dtype + square + incx_pos ],
    [ 'larf',  dtype + mn     + incx + side ],
    [ 'larfx', dtype + mn     + side ],
    [ 'larfb', dtype + mnk    + side + trans + direct + storev ],
    [ 'larft', dtype + nk     + direct + storev ],
    ]

# auxilary - norms
if (args.aux):
    cmds += [
    [ 'lange', dtype + mn +     norm ],
    [ 'lanhe', dtype + square + norm + uplo ],
    [ 'lansy', dtype + square + norm + uplo ],
    [ 'lantr', dtype + square + norm + uplo + diag ],
    ]

# additional blas
if (args.blas):
    cmds += [
    [ 'syr',   dtype + square + uplo ],
    ]

# ------------------------------------------------------------------------------
# when output is redirected to file instead of TTY console,
# print extra messages to stderr on TTY console.
output_redirected = not sys.stdout.isatty()

# ------------------------------------------------------------------------------
def run_test( cmd ):
    cmd = './test ' + ' '.join( cmd )
    print( cmd, file=sys.stderr )
    err = os.system( cmd )
    if (err):
        hi = (err & 0xff00) >> 8
        lo = (err & 0x00ff)
        if (lo == 2):
            print( '\nCancelled', file=sys.stderr )
            exit(1)
        elif (lo != 0):
            print( 'FAILED: abnormal exit, signal =', lo, file=sys.stderr )
        elif (output_redirected):
            print( hi, 'tests FAILED.', file=sys.stderr )
    # end
    return err
# end

# ------------------------------------------------------------------------------
failures = []
run_all = (len(args.tests) == 0)
for cmd in cmds:
    if (run_all or cmd[0] in args.tests):
        err = run_test( cmd )
        if (err != 0):
            failures.append( cmd[0] )

# print summary of failures
nfailures = len( failures )
if (nfailures > 0):
    print( '\n' + str(nfailures) + ' routines FAILED:', ', '.join( failures ),
           file=sys.stderr )
