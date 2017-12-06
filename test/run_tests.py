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
# with single, double and default sizes
#     ./run_tests.py --lu --chol --type s,d
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

group_size = parser.add_argument_group( 'matrix dimensions (default is medium)' )
group_size.add_argument( '-x', '--xsmall', action='store_true', help='run x-small tests' )
group_size.add_argument( '-s', '--small',  action='store_true', help='run small tests' )
group_size.add_argument( '-m', '--medium', action='store_true', help='run medium tests' )
group_size.add_argument( '-l', '--large',  action='store_true', help='run large tests' )
group_size.add_argument(       '--dim',    action='store',      help='explicitly specify size', default='' )

group_cat = parser.add_argument_group( 'category (default is all)' )
categories = [
    group_cat.add_argument( '--lu',            action='store_true', help='run LU tests' ),
    group_cat.add_argument( '--chol',          action='store_true', help='run Cholesky tests' ),
    group_cat.add_argument( '--sysv',          action='store_true', help='run symmetric indefinite (Bunch-Kaufman) tests' ),
    group_cat.add_argument( '--rook',          action='store_true', help='run symmetric indefinite (rook) tests' ),
    group_cat.add_argument( '--aasen',         action='store_true', help='run symmetric indefinite (Aasen) tests' ),
    group_cat.add_argument( '--least-squares', action='store_true', help='run least squares tests' ),
    group_cat.add_argument( '--qr',            action='store_true', help='run QR tests' ),
    group_cat.add_argument( '--lq',            action='store_true', help='run LQ tests' ),
    group_cat.add_argument( '--ql',            action='store_true', help='run QL tests' ),
    group_cat.add_argument( '--rq',            action='store_true', help='run RQ tests' ),
    group_cat.add_argument( '--syev',          action='store_true', help='run symmetric eigenvalues tests' ),
    group_cat.add_argument( '--geev',          action='store_true', help='run non-symmetric eigenvalues tests' ),
    group_cat.add_argument( '--svd',           action='store_true', help='run svd tests' ),
    group_cat.add_argument( '--aux',           action='store_true', help='run auxiliary tests' ),
    group_cat.add_argument( '--aux-house',     action='store_true', help='run auxiliary Householder tests' ),
    group_cat.add_argument( '--aux-norm',      action='store_true', help='run auxiliary norm tests' ),
    group_cat.add_argument( '--blas',          action='store_true', help='run additional BLAS tests' ),
]
categories = map( lambda x: x.dest, categories ) # map to names: ['lu', 'chol', ...]

group_opt = parser.add_argument_group( 'options' )
# BLAS and LAPACK
group_opt.add_argument( '--type',   action='store', help='default=%(default)s', default='s,d,c,z' )
group_opt.add_argument( '--layout', action='store', help='default=%(default)s', default='c,r' )
group_opt.add_argument( '--transA', action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--transB', action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--trans',  action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--uplo',   action='store', help='default=%(default)s', default='l,u' )
group_opt.add_argument( '--diag',   action='store', help='default=%(default)s', default='n,u' )
group_opt.add_argument( '--side',   action='store', help='default=%(default)s', default='l,r' )
group_opt.add_argument( '--incx',   action='store', help='default=%(default)s', default='1,2,-1,-2' )
group_opt.add_argument( '--incy',   action='store', help='default=%(default)s', default='1,2,-1,-2' )
group_opt.add_argument( '--align',  action='store', help='default=%(default)s', default='32' )
# LAPACK only
group_opt.add_argument( '--direct', action='store', help='default=%(default)s', default='f,b' )
group_opt.add_argument( '--storev', action='store', help='default=%(default)s', default='c,r' )
group_opt.add_argument( '--norm',   action='store', help='default=%(default)s', default='max,1,inf,fro' )
group_opt.add_argument( '--matrixtype', action='store', help='default=%(default)s', default='g,l,u' )

parser.add_argument( 'tests', nargs=argparse.REMAINDER )
opts = parser.parse_args()

# by default, run medium sizes
if (not (opts.xsmall or opts.small or opts.medium or opts.large)):
    opts.medium = True

# by default, run all categories
if (opts.tests or not any( map( lambda c: opts.__dict__[ c ], categories ))):
    for c in categories:
        opts.__dict__[ c ] = True

# ------------------------------------------------------------------------------
# parameters
# begin with space to ease concatenation

# if given, use explicit dim
dim = ' --dim ' + opts.dim if (opts.dim) else ''
n        = dim
tall     = dim
wide     = dim
mn       = dim
mnk      = dim
nk_tall  = dim
nk_wide  = dim
nk       = dim

if (not opts.dim):
    if (opts.xsmall):
        n       += ' --dim 10'
        tall    += ' --dim 20x10'
        wide    += ' --dim 10x20'
        mnk     += ' --dim 10x15x20 --dim 15x10x20' \
                +  ' --dim 10x20x15 --dim 15x20x10' \
                +  ' --dim 20x10x15 --dim 20x15x10'
        nk_tall += ' --dim 1x20x10'
        nk_wide += ' --dim 1x10x20'

    if (opts.small):
        n       += ' --dim 25:100:25'
        tall    += ' --dim 50:200:50x25:100:25'  # 2:1
        wide    += ' --dim 25:100:25x50:200:50'  # 1:2
        mnk     += ' --dim 25x50x75 --dim 50x25x75' \
                +  ' --dim 25x75x50 --dim 50x75x25' \
                +  ' --dim 75x25x50 --dim 75x50x25'
        nk_tall += ' --dim 1x50:200:50x25:100:25'
        nk_wide += ' --dim 1x25:100:25x50:200:50'

    if (opts.medium):
        n       += ' --dim 100:500:100'
        tall    += ' --dim 200:1000:200x100:500:100'  # 2:1
        wide    += ' --dim 100:500:100x200:1000:200'  # 1:2
        mnk     += ' --dim 100x300x600 --dim 300x100x600' \
                +  ' --dim 100x600x300 --dim 300x600x100' \
                +  ' --dim 600x100x300 --dim 600x300x100'
        nk_tall += ' --dim 1x200:1000:200x100:500:100'
        nk_wide += ' --dim 1x100:500:100x200:1000:200'

    if (opts.large):
        n       += ' --dim 1000:5000:1000'
        tall    += ' --dim 2000:10000:2000x1000:5000:1000'  # 2:1
        wide    += ' --dim 1000:5000:1000x2000:10000:2000'  # 1:2
        mnk     += ' --dim 1000x3000x6000 --dim 3000x1000x6000' \
                +  ' --dim 1000x6000x3000 --dim 3000x6000x1000' \
                +  ' --dim 6000x1000x3000 --dim 6000x3000x1000'
        nk_tall += ' --dim 1x2000:10000:2000x1000:5000:1000'
        nk_wide += ' --dim 1x1000:5000:1000x2000:10000:2000'

    mn  = n + tall + wide
    mnk = mn + mnk
    nk  = n + nk_tall + nk_wide
# end

# BLAS and LAPACK
dtype  = ' --type '   + opts.type   if (opts.type)   else ''
layout = ' --layout ' + opts.layout if (opts.layout) else ''
transA = ' --transA ' + opts.transA if (opts.transA) else ''
transB = ' --transB ' + opts.transB if (opts.transB) else ''
trans  = ' --trans '  + opts.trans  if (opts.trans)  else ''
uplo   = ' --uplo '   + opts.uplo   if (opts.uplo)   else ''
diag   = ' --diag '   + opts.diag   if (opts.diag)   else ''
side   = ' --side '   + opts.side   if (opts.side)   else ''
incx   = ' --incx '   + opts.incx   if (opts.incx)   else ''
incy   = ' --incy '   + opts.incy   if (opts.incy)   else ''
align  = ' --align '  + opts.align  if (opts.align)  else ''
# LAPACK only
direct = ' --direct ' + opts.direct if (opts.direct) else ''
storev = ' --storev ' + opts.storev if (opts.storev) else ''
norm   = ' --norm '   + opts.norm   if (opts.norm)   else ''
mtype  = ' --matrixtype ' + opts.matrixtype if (opts.matrixtype) else ''

# ------------------------------------------------------------------------------
# filters a comma separated list csv based on items in list values.
# if no items from csv are in values, returns first item in values.
def filter_csv( values, csv ):
    f = filter( lambda x: x in values, csv.split( ',' ))
    if (not f):
        return values[0]
    return ','.join( f )
# end

# ------------------------------------------------------------------------------
# limit options to specific values
dtype_real    = ' --type ' + filter_csv( ('s', 'd'), opts.type )
dtype_complex = ' --type ' + filter_csv( ('c', 'z'), opts.type )

trans_nt = ' --trans ' + filter_csv( ('n', 't'), opts.trans )
trans_nc = ' --trans ' + filter_csv( ('n', 'c'), opts.trans )

# positive inc
incx_pos = ' --incx ' + filter_csv( ('1', '2'), opts.incx )
incy_pos = ' --incy ' + filter_csv( ('1', '2'), opts.incy )

# ------------------------------------------------------------------------------
cmds = []

# LU
if (opts.lu):
    cmds += [
    [ 'gesv',  dtype + align + n ],
    [ 'getrf', dtype + align + mn ],
    [ 'getrs', dtype + align + n + trans ],
    [ 'getri', dtype + align + n ],
    [ 'gecon', dtype + align + n ],
    [ 'gerfs', dtype + align + n + trans ],
    [ 'geequ', dtype + align + n ],
    ]

# Cholesky
if (opts.chol):
    cmds += [
    [ 'posv',  dtype + align + n + uplo ],
    [ 'potrf', dtype + align + n + uplo ],
    [ 'potrs', dtype + align + n + uplo ],
    [ 'potri', dtype + align + n + uplo ],
    [ 'pocon', dtype + align + n + uplo ],
    [ 'porfs', dtype + align + n + uplo ],
    [ 'poequ', dtype + align + n ],  # only diagonal elements (no uplo)
    ]

# symmetric indefinite, Bunch-Kaufman
if (opts.sysv):
    cmds += [
    [ 'sysv',  dtype + align + n + uplo ],
    [ 'sytrf', dtype + align + n + uplo ],
    [ 'sytrs', dtype + align + n + uplo ],
    [ 'sytri', dtype + align + n + uplo ],
    [ 'sycon', dtype + align + n + uplo ],
    [ 'syrfs', dtype + align + n + uplo ],
    ]

# symmetric indefinite, rook
#if (opts.rook):
#    cmds += [
#    [ 'sysv_rook',  dtype + align + n + uplo ],
#    [ 'sytrf_rook', dtype + align + n + uplo ],
#    [ 'sytrs_rook', dtype + align + n + uplo ],
#    [ 'sytri_rook', dtype + align + n + uplo ],
#    ]

# symmetric indefinite, Aasen
#if (opts.aasen):
#    cmds += [
#    [ 'sysv_aasen',  dtype + align + n + uplo ],
#    [ 'sytrf_aasen', dtype + align + n + uplo ],
#    [ 'sytrs_aasen', dtype + align + n + uplo ],
#    [ 'sytri_aasen', dtype + align + n + uplo ],
#    [ 'sysv_aasen_2stage',  dtype + align + n + uplo ],
#    [ 'sytrf_aasen_2stage', dtype + align + n + uplo ],
#    [ 'sytrs_aasen_2stage', dtype + align + n + uplo ],
#    [ 'sytri_aasen_2stage', dtype + align + n + uplo ],
#    ]

# least squares
#if (opts.least_squares):
#    cmds += [
#    [ 'gels',   dtype + align + mn ],
#    [ 'gelsy',  dtype + align + mn ],
#    [ 'gelsd',  dtype + align + mn ],
#    [ 'gelss',  dtype + align + mn ],
#    [ 'getsls', dtype + align + mn ],
#    ]

# QR
#if (opts.qr):
#    cmds += [
#    [ 'geqrf', dtype + align + mn ],
#    [ 'ggqrf', dtype + align + mn ],
#    [ 'ungqr', dtype + align + mn ],
#    [ 'unmqr', dtype + align + mn ],
#    ]

# LQ
#if (opts.lq):
#    cmds += [
#    [ 'gelqf', dtype + align + mn ],
#    [ 'gglqf', dtype + align + mn ],
#    [ 'unglq', dtype + align + mn ],
#    [ 'unmlq', dtype + align + mn ],
#    ]

# QL
#if (opts.ql):
#    cmds += [
#    [ 'geqlf', dtype + align + mn ],
#    [ 'ggqlf', dtype + align + mn ],
#    [ 'ungql', dtype + align + mn ],
#    [ 'unmql', dtype + align + mn ],
#    ]

# RQ
#if (opts.rq):
#    cmds += [
#    [ 'gerqf', dtype + align + mn ],
#    [ 'ggrqf', dtype + align + mn ],
#    [ 'ungrq', dtype + align + mn ],
#    [ 'unmrq', dtype + align + mn ],
#    ]

# symmetric eigenvalues
# todo: add jobs
#if (opts.syev):
#    cmds += [
#    [ 'syev',  dtype + align + n + uplo ],
#    [ 'syevx', dtype + align + n + uplo ],
#    [ 'syevd', dtype + align + n + uplo ],
#    [ 'syevr', dtype + align + n + uplo ],
#    [ 'sytrd', dtype + align + n + uplo ],
#    [ 'orgtr', dtype + align + n + uplo ],
#    [ 'ormtr', dtype + align + n + uplo ],
#    ]

# generalized symmetric eigenvalues
# todo: add jobs
#if (opts.sygv):
#    cmds += [
#    [ 'sygv',  dtype + align + n + uplo ],
#    [ 'sygvx', dtype + align + n + uplo ],
#    [ 'sygvd', dtype + align + n + uplo ],
#    [ 'sygvr', dtype + align + n + uplo ],
#    [ 'sygst', dtype + align + n + uplo ],
#    ]

# non-symmetric eigenvalues
# todo: add jobs
#if (opts.syev):
#    cmds += [
#    [ 'syev',  dtype + align + mn ],
#    [ 'syevx', dtype + align + mn ],
#    [ 'syevd', dtype + align + mn ],
#    [ 'syevr', dtype + align + mn ],
#    [ 'sytrd', dtype + align + mn ],
#    [ 'orgtr', dtype + align + mn ],
#    [ 'ormtr', dtype + align + mn ],
#    ]

# svd
# todo: add jobs
#if (opts.svd):
#    cmds += [
#    [ 'gesvd',         dtype + align + mn ],
#    [ 'gesdd',         dtype + align + mn ],
#    [ 'gesvdx',        dtype + align + mn ],
#    [ 'gesvd_2stage',  dtype + align + mn ],
#    [ 'gesdd_2stage',  dtype + align + mn ],
#    [ 'gesvdx_2stage', dtype + align + mn ],
#    [ 'gejsv',         dtype + align + mn ],
#    [ 'gesvj',         dtype + align + mn ],
#    ]

# auxilary
if (opts.aux):
    cmds += [
    [ 'lacpy', dtype + align + mn + mtype ],
    [ 'laset', dtype + align + mn + mtype ],
    [ 'laswp', dtype + align + mn ],
    ]

# auxilary - householder
if (opts.aux_house):
    cmds += [
    [ 'larfg', dtype         + n   + incx_pos ],
    [ 'larf',  dtype + align + mn  + incx + side ],
    [ 'larfx', dtype + align + mn  + side ],
    [ 'larfb', dtype + align + mnk + side + trans + direct + storev ],
    [ 'larft', dtype + align + nk  + direct + storev ],
    ]

# auxilary - norms
if (opts.aux):
    cmds += [
    [ 'lange', dtype + align + mn + norm ],
    [ 'lanhe', dtype + align + n  + norm + uplo ],
    [ 'lansy', dtype + align + n  + norm + uplo ],
    [ 'lantr', dtype + align + n  + norm + uplo + diag ],
    ]

# additional blas
if (opts.blas):
    cmds += [
    [ 'syr',   dtype + align + n + uplo ],
    ]

# ------------------------------------------------------------------------------
# when output is redirected to file instead of TTY console,
# print extra messages to stderr on TTY console.
output_redirected = not sys.stdout.isatty()

# ------------------------------------------------------------------------------
def run_test( cmd ):
    cmd = './test %-6s%s' % tuple(cmd)
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
run_all = (len(opts.tests) == 0)
for cmd in cmds:
    if (run_all or cmd[0] in opts.tests):
        err = run_test( cmd )
        if (err != 0):
            failures.append( cmd[0] )

# print summary of failures
nfailures = len( failures )
if (nfailures > 0):
    print( '\n' + str(nfailures) + ' routines FAILED:', ', '.join( failures ),
           file=sys.stderr )
