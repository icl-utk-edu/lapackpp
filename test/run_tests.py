#!/usr/bin/env python3
#
# Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
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
import subprocess
import xml.etree.ElementTree as ET
import io
import time

# ------------------------------------------------------------------------------
# command line arguments
parser = argparse.ArgumentParser()

group_test = parser.add_argument_group( 'test' )
group_test.add_argument( '-t', '--test', action='store',
    help='test command to run, e.g., --test "mpirun -np 4 ./test"; default "%(default)s"',
    default='./tester' )
group_test.add_argument( '--xml', help='generate report.xml for jenkins' )
group_test.add_argument( '--dry-run', action='store_true', help='print commands, but do not execute them' )
group_test.add_argument( '--start',   action='store', help='routine to start with, helpful for restarting', default='' )
group_test.add_argument( '-x', '--exclude', action='append', help='routines to exclude; repeatable', default=[] )

group_size = parser.add_argument_group( 'matrix dimensions (default is medium)' )
group_size.add_argument( '--quick',  action='store_true', help='run quick "sanity check" of few, small tests' )
group_size.add_argument( '--xsmall', action='store_true', help='run x-small tests' )
group_size.add_argument( '--small',  action='store_true', help='run small tests' )
group_size.add_argument( '--medium', action='store_true', help='run medium tests' )
group_size.add_argument( '--large',  action='store_true', help='run large tests' )
group_size.add_argument( '--square', action='store_true', help='run square (m = n = k) tests', default=False )
group_size.add_argument( '--tall',   action='store_true', help='run tall (m > n) tests', default=False )
group_size.add_argument( '--wide',   action='store_true', help='run wide (m < n) tests', default=False )
group_size.add_argument( '--mnk',    action='store_true', help='run tests with m, n, k all different', default=False )
group_size.add_argument( '--dim',    action='store',      help='explicitly specify size', default='' )

group_cat = parser.add_argument_group( 'category (default is all)' )
categories = [
    group_cat.add_argument( '--lu',            action='store_true', help='run LU tests' ),
    group_cat.add_argument( '--gb',            action='store_true', help='run GB tests' ),
    group_cat.add_argument( '--gt',            action='store_true', help='run GT tests' ),
    group_cat.add_argument( '--chol',          action='store_true', help='run Cholesky tests' ),
    group_cat.add_argument( '--sysv',          action='store_true', help='run symmetric indefinite (Bunch-Kaufman) tests' ),
    group_cat.add_argument( '--rook',          action='store_true', help='run symmetric indefinite (rook) tests' ),
    group_cat.add_argument( '--aasen',         action='store_true', help='run symmetric indefinite (Aasen) tests' ),
    group_cat.add_argument( '--hesv',          action='store_true', help='run hermetian tests (FIXME more informationhere)' ),
    group_cat.add_argument( '--least-squares', action='store_true', help='run least squares tests' ),
    group_cat.add_argument( '--qr',            action='store_true', help='run QR tests' ),
    group_cat.add_argument( '--lq',            action='store_true', help='run LQ tests' ),
    group_cat.add_argument( '--ql',            action='store_true', help='run QL tests' ),
    group_cat.add_argument( '--rq',            action='store_true', help='run RQ tests' ),
    group_cat.add_argument( '--syev',          action='store_true', help='run symmetric eigenvalues tests' ),
    group_cat.add_argument( '--sygv',          action='store_true', help='run generalized symmetric eigenvalues tests' ),
    group_cat.add_argument( '--geev',          action='store_true', help='run non-symmetric eigenvalues tests' ),
    group_cat.add_argument( '--svd',           action='store_true', help='run svd tests' ),
    group_cat.add_argument( '--aux',           action='store_true', help='run auxiliary tests' ),
    group_cat.add_argument( '--aux-house',     action='store_true', help='run auxiliary Householder tests' ),
    group_cat.add_argument( '--aux-norm',      action='store_true', help='run auxiliary norm tests' ),
    group_cat.add_argument( '--blas',          action='store_true', help='run additional BLAS tests' ),

]
# map category objects to category names: ['lu', 'chol', ...]
categories = list( map( lambda x: x.dest, categories ) )

group_target = parser.add_argument_group( 'target' )
group_target.add_argument( '--host',          action='store_true', help='run all CPU host routines' ),
group_target.add_argument( '--device',        action='store_true', help='run all GPU device routines' ),

group_opt = parser.add_argument_group( 'options' )
# BLAS and LAPACK
# Empty defaults (check, ref, etc.) use the default in test.cc.
group_opt.add_argument( '--type',   action='store', help='default=%(default)s', default='s,d,c,z' )
group_opt.add_argument( '--layout', action='store', help='default=%(default)s', default='c,r' )
group_opt.add_argument( '--transA', action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--transB', action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--trans',  action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--uplo',   action='store', help='default=%(default)s', default='l,u' )
group_opt.add_argument( '--diag',   action='store', help='default=%(default)s', default='n,u' )
group_opt.add_argument( '--side',   action='store', help='default=%(default)s', default='l,r' )
group_opt.add_argument( '--alpha',  action='store', help='default=%(default)s', default='' )
group_opt.add_argument( '--beta',   action='store', help='default=%(default)s', default='' )
group_opt.add_argument( '--incx',   action='store', help='default=%(default)s', default='1,2,-1,-2' )
group_opt.add_argument( '--incy',   action='store', help='default=%(default)s', default='1,2,-1,-2' )
group_opt.add_argument( '--align',  action='store', help='default=%(default)s', default='32' )
group_opt.add_argument( '--check',  action='store', help='default=y', default='' )  # default in test.cc
group_opt.add_argument( '--ref',    action='store', help='default=y', default='' )  # default in test.cc
group_opt.add_argument( '--verbose', action='store', help='default=0', default='' )  # default in test.cc

# LAPACK only
group_opt.add_argument( '--itype',  action='store', help='default=%(default)s', default='1,2,3' )
group_opt.add_argument( '--factored', action='store', help='default=%(default)s', default='f,n,e' )
group_opt.add_argument( '--equed',  action='store', help='default=%(default)s', default='n,r,c,b' )
group_opt.add_argument( '--direction', action='store', help='default=%(default)s', default='f,b' )
group_opt.add_argument( '--storev', action='store', help='default=%(default)s', default='c,r' )
group_opt.add_argument( '--norm',   action='store', help='default=%(default)s', default='max,1,inf,fro' )
group_opt.add_argument( '--ijob',   action='store', help='default=%(default)s', default='0:5:1' )
group_opt.add_argument( '--jobz',   action='store', help='default=%(default)s', default='n,v' )
group_opt.add_argument( '--jobvl',  action='store', help='default=%(default)s', default='n,v' )
group_opt.add_argument( '--jobvr',  action='store', help='default=%(default)s', default='n,v' )
group_opt.add_argument( '--jobvs',  action='store', help='default=%(default)s', default='n,v' )
group_opt.add_argument( '--jobu',   action='store', help='default=%(default)s', default='n,s,o,a' )
group_opt.add_argument( '--jobvt',  action='store', help='default=%(default)s', default='n,s,o,a' )
group_opt.add_argument( '--balanc', action='store', help='default=%(default)s', default='n,p,s,b' )
group_opt.add_argument( '--sort',   action='store', help='default=%(default)s', default='n,s' )
group_opt.add_argument( '--select', action='store', help='default=%(default)s', default='n,s' )
group_opt.add_argument( '--sense',  action='store', help='default=%(default)s', default='n,e,v,b' )
group_opt.add_argument( '--vect',   action='store', help='default=%(default)s', default='n,v' )
group_opt.add_argument( '--l',      action='store', help='default=%(default)s', default='0,100' )
group_opt.add_argument( '--ka',     action='store', help='default=%(default)s', default='20,100' )
group_opt.add_argument( '--kb',     action='store', help='default=%(default)s', default='20,100' )
group_opt.add_argument( '--kd',     action='store', help='default=%(default)s', default='20,100' )
group_opt.add_argument( '--kl',     action='store', help='default=%(default)s', default='20,100' )
group_opt.add_argument( '--ku',     action='store', help='default=%(default)s', default='20,100' )
group_opt.add_argument( '--vl',     action='store', help='default=%(default)s', default='-inf,0' )
group_opt.add_argument( '--vu',     action='store', help='default=%(default)s', default='inf' )
group_opt.add_argument( '--il',     action='store', help='default=%(default)s', default='10' )
group_opt.add_argument( '--iu',     action='store', help='default=%(default)s', default='-1,100' )
group_opt.add_argument( '--nb',     action='store', help='default=%(default)s', default='64' )
group_opt.add_argument( '--matrixtype', action='store', help='default=%(default)s', default='g,l,u' )

parser.add_argument( 'tests', nargs=argparse.REMAINDER )
opts = parser.parse_args()

for t in opts.tests:
    if (t.startswith('--')):
        print( 'Error: option', t, 'must come before any routine names' )
        print( 'usage:', sys.argv[0], '[options]', '[routines]' )
        print( '      ', sys.argv[0], '--help' )
        exit(1)

# by default, run medium sizes
if (not (opts.quick or opts.xsmall or opts.small or opts.medium or opts.large)):
    opts.medium = True

# by default, run all shapes
if (not (opts.square or opts.tall or opts.wide or opts.mnk)):
    opts.square = True
    opts.tall   = True
    opts.wide   = True
    opts.mnk    = True

# By default, run both host and device.
if (not opts.host and not opts.device):
    opts.host   = True
    opts.device = True

# By default, or if specific test routines given, enable all categories
# to get whichever has the routines.
if (opts.tests or not any( map( lambda c: opts.__dict__[ c ], categories ))):
    for c in categories:
        opts.__dict__[ c ] = True

start_routine = opts.start

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
    if (opts.quick):
        n        = ' --dim 100'
        tall     = ' --dim 100x50'  # 2:1
        wide     = ' --dim 50x100'  # 1:2
        mnk      = ' --dim 25x50x75'
        nk_tall  = ' --dim 1x100x50'  # 2:1
        nk_wide  = ' --dim 1x50x100'  # 1:2
        opts.incx  = '1,-1'
        opts.incy  = '1,-1'
        opts.batch = '10'
        opts.l     = '0,20,50'
        opts.nb    = '16'

    if (opts.xsmall):
        n       += ' --dim 10'
        tall    += ' --dim 20x10'
        wide    += ' --dim 10x20'
        mnk     += ' --dim 10x15x20 --dim 15x10x20' \
                +  ' --dim 10x20x15 --dim 15x20x10' \
                +  ' --dim 20x10x15 --dim 20x15x10'
        nk_tall += ' --dim 1x20x10'
        nk_wide += ' --dim 1x10x20'
        # tpqrt, tplqt needs small l, nb <= min( m, n )
        if (opts.l == parser.get_default('l')):
            opts.l = '0,5,100'
        if (opts.nb == parser.get_default('nb')):
            opts.nb = '8,64'
        if (opts.ka == parser.get_default('ka')):
            opts.ka = '5'
        if (opts.kb == parser.get_default('kb')):
            opts.kb = '5'
        if (opts.kd == parser.get_default('kd')):
            opts.kd = '5'

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

    mn  = ''
    nk  = ''
    if (opts.square):
        mn = n
        nk = n
    if (opts.tall):
        mn += tall
        nk += nk_tall
    if (opts.wide):
        mn += wide
        nk += nk_wide
    if (opts.mnk):
        mnk = mn + mnk
    else:
        mnk = mn
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
a      = ' --alpha '  + opts.alpha  if (opts.alpha)  else ''
ab     = a+' --beta ' + opts.beta   if (opts.beta)   else a
incx   = ' --incx '   + opts.incx   if (opts.incx)   else ''
incy   = ' --incy '   + opts.incy   if (opts.incy)   else ''
align  = ' --align '  + opts.align  if (opts.align)  else ''
check  = ' --check '  + opts.check  if (opts.check)  else ''
ref    = ' --ref '    + opts.ref    if (opts.ref)    else ''
verbose = ' --verbose ' + opts.verbose if (opts.verbose) else ''

# LAPACK only
itype  = ' --itype '  + opts.itype  if (opts.itype)  else ''
factored = ' --factored ' + opts.factored if (opts.factored)  else ''
equed  = ' --equed '  + opts.equed  if (opts.equed)  else ''
direction = ' --direction ' + opts.direction if (opts.direction) else ''
storev = ' --storev ' + opts.storev if (opts.storev) else ''
norm   = ' --norm '   + opts.norm   if (opts.norm)   else ''
ijob   = ' --ijob '   + opts.ijob   if (opts.ijob)   else ''
jobz   = ' --jobz '   + opts.jobz   if (opts.jobz)   else ''
jobu   = ' --jobu '   + opts.jobu   if (opts.jobu)   else ''
jobvt  = ' --jobvt '  + opts.jobvt  if (opts.jobvt)  else ''
jobvl  = ' --jobvl '  + opts.jobvl  if (opts.jobvl)  else ''
jobvr  = ' --jobvr '  + opts.jobvr  if (opts.jobvr)  else ''
jobvs  = ' --jobvs '  + opts.jobvs  if (opts.jobvs)  else ''
balanc = ' --balanc ' + opts.balanc if (opts.balanc) else ''
sort   = ' --sort '   + opts.sort   if (opts.sort)   else ''
sense  = ' --sense '  + opts.sense  if (opts.sense)  else ''
vect   = ' --vect '   + opts.vect   if (opts.vect)   else ''
l      = ' --l '      + opts.l      if (opts.l)      else ''
nb     = ' --nb '     + opts.nb     if (opts.nb)     else ''
ka     = ' --ka '     + opts.ka     if (opts.ka)     else ''
kb     = ' --kb '     + opts.kb     if (opts.kb)     else ''
kd     = ' --kd '     + opts.kd     if (opts.kd)     else ''
kl     = ' --kl '     + opts.kl     if (opts.kl)     else ''
ku     = ' --ku '     + opts.ku     if (opts.ku)     else ''
vl     = ' --vl '     + opts.vl     if (opts.vl)     else ''
vu     = ' --vu '     + opts.vu     if (opts.vu)     else ''
il     = ' --il '     + opts.il     if (opts.il)     else ''
iu     = ' --iu '     + opts.iu     if (opts.iu)     else ''
mtype  = ' --matrixtype ' + opts.matrixtype if (opts.matrixtype) else ''

# general options for all routines
gen = check + ref + verbose

# ------------------------------------------------------------------------------
# filters a comma separated list csv based on items in list values.
# if no items from csv are in values, returns first item in values.
def filter_csv( values, csv ):
    f = list( filter( lambda x: x in values, csv.split( ',' ) ) )
    if (not f):
        return values[0]
    return ','.join( f )
# end

# ------------------------------------------------------------------------------
# limit options to specific values
dtype_real    = ' --type ' + filter_csv( ('s', 'd'), opts.type )
dtype_complex = ' --type ' + filter_csv( ('c', 'z'), opts.type )
dtype_double  = ' --type ' + filter_csv( ('d', 'z'), opts.type )

trans_nt = ' --trans ' + filter_csv( ('n', 't'), opts.trans )
trans_nc = ' --trans ' + filter_csv( ('n', 'c'), opts.trans )

# positive inc
incx_pos = ' --incx ' + filter_csv( ('1', '2'), opts.incx )
incy_pos = ' --incy ' + filter_csv( ('1', '2'), opts.incy )

# ------------------------------------------------------------------------------
cmds = []

# LU
if (opts.lu and opts.host):
    cmds += [
    [ 'gesv',  gen + dtype + align + n ],
    # todo: equed
    [ 'gesvx', gen + dtype + align + n + factored + trans ],
    [ 'getrf', gen + dtype + align + mn ],
    [ 'getrs', gen + dtype + align + n + trans ],
    [ 'getri', gen + dtype + align + n ],
    [ 'gecon', gen + dtype + align + n ],
    [ 'gerfs', gen + dtype + align + n + trans ],
    [ 'geequ', gen + dtype + align + n ],
    ]

if (opts.lu and opts.device):
    # GPU
    cmds += [
    [ 'dev-getrf', gen + dtype + align + n ],
    ]

# General Banded
if (opts.gb and opts.host):
    cmds += [
    [ 'gbsv',  gen + dtype + align + n  + kl + ku ],
    [ 'gbtrf', gen + dtype + align + mn + kl + ku ],
    [ 'gbtrs', gen + dtype + align + n  + kl + ku + trans ],
    [ 'gbcon', gen + dtype + align + n  + kl + ku ],
    [ 'gbrfs', gen + dtype + align + n  + kl + ku + trans ],
    [ 'gbequ', gen + dtype + align + n  + kl + ku ],
    ]

# General Tri-Diagonal
if (opts.gt and opts.host):
    cmds += [
    [ 'gtsv',  gen + dtype + align + n ],
    [ 'gttrf', gen + dtype +         n ],
    [ 'gttrs', gen + dtype + align + n + trans ],
    [ 'gtcon', gen + dtype +         n ],
    [ 'gtrfs', gen + dtype + align + n + trans ],
    ]

# Cholesky
if (opts.chol and opts.host):
    cmds += [
    [ 'posv',  gen + dtype + align + n + uplo ],
    [ 'potrf', gen + dtype + align + n + uplo ],
    [ 'potrs', gen + dtype + align + n + uplo ],
    [ 'potri', gen + dtype + align + n + uplo ],
    [ 'pocon', gen + dtype + align + n + uplo ],
    [ 'porfs', gen + dtype + align + n + uplo ],
    [ 'poequ', gen + dtype + align + n ],  # only diagonal elements (no uplo)

    # Packed
    [ 'ppsv',  gen + dtype + align + n + uplo ],
    [ 'pptrf', gen + dtype +         n + uplo ],
    [ 'pptrs', gen + dtype + align + n + uplo ],
    [ 'pptri', gen + dtype +         n + uplo ],
    [ 'ppcon', gen + dtype +         n + uplo ],
    [ 'pprfs', gen + dtype + align + n + uplo ],
    [ 'ppequ', gen + dtype +         n + uplo ],

    # Banded
    [ 'pbsv',  gen + dtype + align + n + kd + uplo ],
    [ 'pbtrf', gen + dtype + align + n + kd + uplo ],
    [ 'pbtrs', gen + dtype + align + n + kd + uplo ],
    [ 'pbcon', gen + dtype + align + n + kd + uplo ],
    [ 'pbrfs', gen + dtype + align + n + kd + uplo ],
    [ 'pbequ', gen + dtype + align + n + kd + uplo ],

    # Tri-diagonal
    [ 'ptsv',  gen + dtype + align + n ],
    [ 'pttrf', gen + dtype         + n ],
    [ 'pttrs', gen + dtype + align + n + uplo ],
    [ 'ptcon', gen + dtype         + n ],
    [ 'ptrfs', gen + dtype + align + n + uplo ],
    ]

if (opts.chol and opts.device):
    # GPU
    cmds += [
    [ 'dev-potrf', gen + dtype + align + n + uplo ],
    ]

# symmetric indefinite, Bunch-Kaufman
if (opts.sysv and opts.host):
    cmds += [
    [ 'sysv',  gen + dtype + align + n + uplo ],
    [ 'sytrf', gen + dtype + align + n + uplo ],
    [ 'sytrs', gen + dtype + align + n + uplo ],
    [ 'sytri', gen + dtype + align + n + uplo ],
    [ 'sycon', gen + dtype + align + n + uplo ],
    [ 'syrfs', gen + dtype + align + n + uplo ],

    # Packed
    [ 'spsv',  gen + dtype + align + n + uplo ],
    [ 'sptrf', gen + dtype         + n + uplo ],
    [ 'sptrs', gen + dtype + align + n + uplo ],
    [ 'sptri', gen + dtype         + n + uplo ],
    [ 'spcon', gen + dtype         + n + uplo ],
    [ 'sprfs', gen + dtype + align + n + uplo ],
    ]

# symmetric indefinite, rook
if (opts.rook and opts.host):
    cmds += [
    # original Rook
    [ 'sysv_rook',  gen + dtype + align + n + uplo ],
    [ 'sytrf_rook', gen + dtype + align + n + uplo ],
    [ 'sytrs_rook', gen + dtype + align + n + uplo ],
    #[ 'sytri_rook', gen + dtype + align + n + uplo ],

    # new Rook
    [ 'sysv_rk',    gen + dtype + align + n + uplo ],
    [ 'sytrf_rk',   gen + dtype + align + n + uplo ],
    #[ 'sytrs_rk',   gen + dtype + align + n + uplo ],
    #[ 'sytri_rk',   gen + dtype + align + n + uplo ],
    ]

# symmetric indefinite, Aasen
if (opts.aasen and opts.host):
    cmds += [
    [ 'sysv_aa',  gen + dtype + align + n + uplo ],
    [ 'sytrf_aa', gen + dtype + align + n + uplo ],
    [ 'sytrs_aa', gen + dtype + align + n + uplo ],
    #[ 'sytri_aa', gen + dtype + align + n + uplo ],

    #[ 'sysv_aasen_2stage',  gen + dtype + align + n + uplo ],
    #[ 'sytrf_aasen_2stage', gen + dtype + align + n + uplo ],
    #[ 'sytrs_aasen_2stage', gen + dtype + align + n + uplo ],
    #[ 'sytri_aasen_2stage', gen + dtype + align + n + uplo ],
    ]

# Hermitian indefinite
if (opts.hesv and opts.host):
    cmds += [
    [ 'hesv',  gen + dtype + align + n + uplo ],
    [ 'hetrf', gen + dtype + align + n + uplo ],
    [ 'hetrs', gen + dtype + align + n + uplo ],
    [ 'hetri', gen + dtype + align + n + uplo ],
    [ 'hecon', gen + dtype + align + n + uplo ],
    [ 'herfs', gen + dtype + align + n + uplo ],

    # Packed
    [ 'hpsv',  gen + dtype + align + n + uplo ],
    [ 'hptrf', gen + dtype + n + uplo ],
    [ 'hptrs', gen + dtype + align + n + uplo ],
    [ 'hptri', gen + dtype + n + uplo ],
    [ 'hpcon', gen + dtype + n + uplo ],
    [ 'hprfs', gen + dtype + align + n + uplo ],
    ]

# least squares
if (opts.least_squares and opts.host):
    cmds += [
    [ 'gels',   gen + dtype + align + mn + trans_nc ],
    [ 'gelsy',  gen + dtype + align + mn ],
    # todo: gelsd is failing
    #[ 'gelsd',  gen + dtype + align + mn ],
    [ 'gelss',  gen + dtype + align + mn ],
    [ 'getsls', gen + dtype + align + mn + trans_nc ],

    # Generalized
    [ 'gglse', gen + dtype + align + mnk ],
    # todo: ggglm is failing
    #[ 'ggglm', gen + dtype + align + mnk ],
    ]

# QR
if (opts.qr and opts.host):
    cmds += [
    [ 'geqr',  gen + dtype + align + n + wide + tall ],
    [ 'geqrf', gen + dtype + align + n + wide + tall ],
    # todo: ggqrf is failing
    #[ 'ggqrf', gen + dtype + align + mnk ],
    [ 'ungqr', gen + dtype + align + mn ],  # m >= n
    #[ 'unmqr', gen + dtype_real    + align + mnk + side + trans    ],  # real does trans = N, T, C
    #[ 'unmqr', gen + dtype_complex + align + mnk + side + trans_nc ],  # complex does trans = N, C, not T

    [ 'orhr_col', gen + dtype_real + align + n + tall ],
    [ 'unhr_col', gen + dtype      + align + n + tall ],

    [ 'gemqrt', gen + dtype_real    + align + n + nb + side + trans    ],  # real does trans = N, T, C
    [ 'gemqrt', gen + dtype_complex + align + n + nb + side + trans_nc ],  # complex does trans = N, C, not T

    # Triangle-pentagon
    [ 'tpqrt',  gen + dtype + align + mn + l + nb ],
    [ 'tpqrt2', gen + dtype + align + mn + l ],
    [ 'tpmqrt', gen + dtype_real    + align + mn + l + nb + side + trans    ],  # real does trans = N, T, C
    [ 'tpmqrt', gen + dtype_complex + align + mn + l + nb + side + trans_nc ],  # complex does trans = N, C, not T
    #[ 'tprfb',  gen + dtype + align + mn + l ],  # TODO: bug in LAPACKE crashes tester
    ]

if (opts.qr and opts.device):
    # GPU
    cmds += [
    [ 'dev-geqrf', gen + dtype + align + n + wide + tall ],
    ]

# LQ
if (opts.lq and opts.host):
    cmds += [
    [ 'gelqf', gen + dtype + align + mn ],
    #[ 'gglqf', gen + dtype + align + mn ],
    [ 'unglq', gen + dtype + align + mn ],  # m <= n, k <= m  TODO Fix the input sizes to match constraints
    #[ 'unmlq', gen + dtype_real    + align + mnk + side + trans    ],  # real does trans = N, T, C
    #[ 'unmlq', gen + dtype_complex + align + mnk + side + trans_nc ],  # complex does trans = N, C, not T

    # Triangle-pentagon
    [ 'tplqt',  gen + dtype + align + mn + l + nb ],
    [ 'tplqt2', gen + dtype + align + mn + l ],
    [ 'tpmlqt', gen + dtype_real    + align + mn + l + nb + side + trans    ],  # real does trans = N, T, C
    [ 'tpmlqt', gen + dtype_complex + align + mn + l + nb + side + trans_nc ],  # complex does trans = N, C, not T
    ]

# QL
if (opts.ql and opts.host):
    cmds += [
    [ 'geqlf', gen + dtype + align + mn ],
    #[ 'ggqlf', gen + dtype + align + mn ],
    [ 'ungql', gen + dtype + align + mn ],
    #[ 'unmql', gen + dtype_real    + align + mnk + side + trans    ],  # real does trans = N, T, C
    #[ 'unmql', gen + dtype_complex + align + mnk + side + trans_nc ],  # complex does trans = N, C, not T
    ]

# RQ
if (opts.rq and opts.host):
    cmds += [
    [ 'gerqf', gen + dtype + align + mn ],
    # todo: ggrqf is failing
    #[ 'ggrqf', gen + dtype + align + mnk ],
    [ 'ungrq', gen + dtype + align + mnk ],
    #[ 'unmrq', gen + dtype_real    + align + mnk + side + trans    ],  # real does trans = N, T, C
    #[ 'unmrq', gen + dtype_complex + align + mnk + side + trans_nc ],  # complex does trans = N, C, not T
    ]

# symmetric eigenvalues
if (opts.syev and opts.host):
    cmds += [
    [ 'heev',  gen + dtype + align + n + jobz + uplo ],
    [ 'heevx', gen + dtype + align + n + jobz + uplo + vl + vu ],
    [ 'heevx', gen + dtype + align + n + jobz + uplo + il + iu ],
    [ 'heevd', gen + dtype + align + n + jobz + uplo ],
    [ 'heevr', gen + dtype + align + n + jobz + uplo + vl + vu ],
    [ 'heevr', gen + dtype + align + n + jobz + uplo + il + iu ],
    [ 'hetrd', gen + dtype + align + n + uplo ],
    [ 'ungtr', gen + dtype + align + n + uplo ],
    [ 'unmtr', gen + dtype_real    + align + mn + uplo + side + trans    ],  # real does trans = N, T, C
    [ 'unmtr', gen + dtype_complex + align + mn + uplo + side + trans_nc ],  # complex does trans = N, C, not T

    # Packed
    [ 'hpev',  gen + dtype + align + n + jobz + uplo ],
    [ 'hpevx', gen + dtype + align + n + jobz + uplo + vl + vu ],
    [ 'hpevx', gen + dtype + align + n + jobz + uplo + il + iu ],
    [ 'hpevd', gen + dtype + align + n + jobz + uplo ],
    #[ 'hpevr', gen + dtype + align + n + jobz + uplo + vl + vu ],
    #[ 'hpevr', gen + dtype + align + n + jobz + uplo + il + iu ],
    [ 'hptrd', gen + dtype + n + uplo ],
    [ 'upgtr', gen + dtype + align + n + uplo ],
    [ 'upmtr', gen + dtype + align + mn + side + uplo + trans_nc ],

    # Banded
    [ 'hbev',  gen + dtype + align + n + jobz + uplo ],
    [ 'hbevx', gen + dtype + align + n + jobz + uplo + vl + vu ],
    [ 'hbevx', gen + dtype + align + n + jobz + uplo + il + iu ],
    [ 'hbevd', gen + dtype + align + n + jobz + uplo ],
    #[ 'hbevr', gen + dtype + align + n + jobz + uplo + vl + vu ],
    #[ 'hbevr', gen + dtype + align + n + jobz + uplo + il + iu ],
    #[ 'hbtrd', gen + dtype + align + n + uplo ],
    #[ 'ubgtr', gen + dtype + align + n + uplo ],
    #[ 'ubmtr', gen + dtype_real    + la + mn + uplo + side + trans    ],
    #[ 'ubmtr', gen + dtype_complex + la + mn + uplo + side + trans_nc ],
    ]

# generalized symmetric eigenvalues
if (opts.sygv and opts.host):
    cmds += [
    [ 'hegv',  gen + dtype + align + n + itype + jobz + uplo ],
    [ 'hegvx', gen + dtype + align + n + itype + jobz + uplo + vl + vu ],
    [ 'hegvx', gen + dtype + align + n + itype + jobz + uplo + il + iu ],
    [ 'hegvd', gen + dtype + align + n + itype + jobz + uplo ],
    #[ 'hegvr', gen + dtype + align + n + uplo ],
    [ 'hegst', gen + dtype + align + n + itype + uplo ],

    # Packed
    [ 'hpgv',  gen + dtype + align + n + itype + jobz + uplo ],
    [ 'hpgvx', gen + dtype + align + n + itype + jobz + uplo + vl + vu ],
    [ 'hpgvx', gen + dtype + align + n + itype + jobz + uplo + il + iu ],
    [ 'hpgvd', gen + dtype + align + n + itype + jobz + uplo ],
    #[ 'hpgvr', gen + dtype + align + n + uplo ],
    [ 'hpgst', gen + dtype + n + itype + uplo ],

    # Banded
    [ 'hbgv',  gen + dtype + align + n + jobz + uplo + ka + kb ],
    [ 'hbgvx', gen + dtype + align + n + jobz + uplo + ka + kb + vl + vu ],
    [ 'hbgvx', gen + dtype + align + n + jobz + uplo + ka + kb + il + iu ],
    [ 'hbgvd', gen + dtype + align + n + jobz + uplo + ka + kb ],
    #[ 'hbgvr', gen + dtype + align + n + uplo + ka + kb ],
    #[ 'hbgst', gen + dtype + align + n + vect + uplo + ka + kb ],
    ]

# non-symmetric eigenvalues
if (opts.geev and opts.host):
    cmds += [
    [ 'geev',  gen + dtype + align + n + jobvl + jobvr ],
    # todo: ggev is failing
    #[ 'ggev',  gen + dtype + align + n + jobvl + jobvr ],
    #[ 'geevx', gen + dtype + align + n + balanc + jobvl + jobvr + sense ],
    [ 'gehrd', gen + dtype + align + n ],
    [ 'unghr', gen + dtype + align + n ],
    [ 'unmhr', gen + dtype_real    + align + mn + side + trans    ],  # real does trans = N, T, C
    [ 'unmhr', gen + dtype_complex + align + mn + side + trans_nc ],  # complex does trans = N, C, not T
    #[ 'trevc', gen + dtype + align + n + side + howmany + select ],
    #[ 'geesx', gen + dtype + align + n + jobvs + sort + select + sense ],
    [ 'tgexc', gen + dtype + align + n + jobvl + jobvr ],
    [ 'tgsen', gen + dtype + align + n + jobvl + jobvr + ijob ],
    ]

# svd
if (opts.svd and opts.host):
    cmds += [
    # todo: MKL seems to have a bug with jobu=o,s and jobvt=o,s,a
    # for tall matrices, e.g., dim=100x50. Skip failing combinations for now.
    #[ 'gesvd',         gen + dtype + align + mn + jobu + jobvt ],
    [ 'gesvd',         gen + dtype + align + mn + " --jobu n,a" + jobvt ],
    [ 'gesvd',         gen + dtype + align + mn + " --jobu o,s --jobvt n" ],
    [ 'gesdd',         gen + dtype + align + mn + jobu ],
    # todo: gesvdx is failing
    #[ 'gesvdx',        gen + dtype + align + mn + jobz + jobvr + vl + vu ],
    #[ 'gesvdx',        gen + dtype + align + mn + jobz + jobvr + il + iu ],
    #[ 'gesvd_2stage',  gen + dtype + align + mn ],
    #[ 'gesdd_2stage',  gen + dtype + align + mn ],
    #[ 'gesvdx_2stage', gen + dtype + align + mn ],
    #[ 'gejsv',         gen + dtype + align + mn ],
    #[ 'gesvj',         gen + dtype + align + mn + joba + jobu + jobv ],
    ]

# auxilary
if (opts.aux and opts.host):
    cmds += [
    [ 'lacpy', gen + dtype + align + mn + mtype ],
    [ 'laed4', gen + dtype_real + n ],
    [ 'laset', gen + dtype + align + mn + mtype ],
    [ 'laswp', gen + dtype + align + mn ],
    ]

# auxilary - householder
if (opts.aux_house and opts.host):
    cmds += [
    [ 'larfg', dtype         + n   + incx_pos ],
    [ 'larfgp', dtype        + n   + incx_pos ],
    [ 'larf',  gen + dtype + align + mn  + incx + side ],
    [ 'larfx', gen + dtype + align + mn  + side ],
    [ 'larfy', gen + dtype + align + n   + incx ],
    [ 'larfb', gen + dtype + align + mnk + side + trans + direction + storev ],
    [ 'larft', gen + dtype + align + nk  + direction + storev ],
    ]

# auxilary - norms
if (opts.aux_norm and opts.host):
    cmds += [
    [ 'lange', gen + dtype + align + mn + norm ],
    # todo: lanhe is failing
    #[ 'lanhe', gen + dtype + align + n  + norm + uplo ],
    [ 'lansy', gen + dtype + align + n  + norm + uplo ],
    [ 'lantr', gen + dtype + align + mn + norm + uplo + diag ],
    [ 'lanhs', gen + dtype + align + n  + norm ],

    # Packed
    [ 'lanhp', gen + dtype + n + norm + uplo ],
    [ 'lansp', gen + dtype + n + norm + uplo ],
    [ 'lantp', gen + dtype + n + norm + uplo + diag ],

    # Banded
    [ 'langb', gen + dtype + align + mn + kl + ku + norm ],
    [ 'lanhb', gen + dtype + align + n + kd + norm + uplo ],
    [ 'lansb', gen + dtype + align + n + kd + norm + uplo ],
    [ 'lantb', gen + dtype + align + n + kd + norm + uplo + diag ],

    # Tri-diagonal
    [ 'langt', gen + dtype + n + norm ],
    [ 'lanht', gen + dtype + n + norm ],
    [ 'lanst', gen + dtype + n + norm ],
    ]

# additional blas
if (opts.blas and opts.host):
    cmds += [
    [ 'syr',   gen + dtype + align + n + uplo ],
    [ 'symv',  gen + dtype + layout + align + uplo + n + incx + incy ],
    ]

# ------------------------------------------------------------------------------
# When stdout is redirected to file instead of TTY console,
# and  stderr is still going to a TTY console,
# print extra summary messages to stderr.
output_redirected = sys.stderr.isatty() and not sys.stdout.isatty()

# ------------------------------------------------------------------------------
# if output is redirected, prints to both stderr and stdout;
# otherwise prints to just stdout.
def print_tee( *args ):
    global output_redirected
    print( *args )
    if (output_redirected):
        print( *args, file=sys.stderr )
# end

# ------------------------------------------------------------------------------
# cmd is a pair of strings: (function, args)

def run_test( cmd ):
    cmd = opts.test +' '+ cmd[1] +' '+ cmd[0]
    print_tee( cmd )
    if (opts.dry_run):
        return (None, None)

    output = ''
    p = subprocess.Popen( cmd.split(), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT )
    p_out = p.stdout
    if (sys.version_info.major >= 3):
        p_out = io.TextIOWrapper(p.stdout, encoding='utf-8')
    # Read unbuffered ("for line in p.stdout" will buffer).
    for line in iter(p_out.readline, ''):
        print( line, end='' )
        output += line
    err = p.wait()
    if (err != 0):
        print_tee( 'FAILED: exit code', err )
    else:
        print_tee( 'pass' )
    return (err, output)
# end

# ------------------------------------------------------------------------------
# Utility to pretty print XML.
# See https://stackoverflow.com/a/33956544/1655607
#
def indent_xml( elem, level=0 ):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent_xml( elem, level+1 )
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
# end

# ------------------------------------------------------------------------------
# run each test

start = time.time()
print_tee( time.ctime() )

failed_tests = []
passed_tests = []
ntests = len(opts.tests)
run_all = (ntests == 0)

seen = set()
for cmd in cmds:
    if ((run_all or cmd[0] in opts.tests) and cmd[0] not in opts.exclude):
        if (start_routine and cmd[0] != start_routine):
            print_tee( 'skipping', cmd[0] )
            continue
        start_routine = None

        seen.add( cmd[0] )
        (err, output) = run_test( cmd )
        if (err):
            failed_tests.append( (cmd[0], err, output) )
        else:
            passed_tests.append( cmd[0] )

not_seen = list( filter( lambda x: x not in seen, opts.tests ) )
if (not_seen):
    print_tee( 'Warning: unknown routines:', ' '.join( not_seen ))

# print summary of failures
nfailed = len( failed_tests )
if (nfailed > 0):
    print_tee( '\n' + str(nfailed) + ' routines FAILED:',
               ', '.join( [x[0] for x in failed_tests] ) )
else:
    print_tee( '\n' + 'All routines passed.' )

# generate jUnit compatible test report
if opts.xml:
    print( 'writing XML file', opts.xml )
    root = ET.Element("testsuites")
    doc = ET.SubElement(root, "testsuite",
                        name="lapackpp_suite",
                        tests=str(ntests),
                        errors="0",
                        failures=str(nfailed))

    for (test, err, output) in failed_tests:
        testcase = ET.SubElement(doc, "testcase", name=test)

        failure = ET.SubElement(testcase, "failure")
        if (err < 0):
            failure.text = "exit with signal " + str(-err)
        else:
            failure.text = str(err) + " tests failed"

        system_out = ET.SubElement(testcase, "system-out")
        system_out.text = output
    # end

    for test in passed_tests:
        testcase = ET.SubElement(doc, 'testcase', name=test)
        testcase.text = 'PASSED'

    tree = ET.ElementTree(root)
    indent_xml( root )
    tree.write( opts.xml )
# end

elapsed = time.time() - start
print_tee( 'Elapsed %.2f sec' % elapsed )
print_tee( time.ctime() )

exit( nfailed )
