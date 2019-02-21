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
import subprocess
import xml.etree.ElementTree as ET

# ------------------------------------------------------------------------------
# found at: https://stackoverflow.com/questions/15203829/python-argparse-file-extension-checking
def check_file_ext(choices):
    class Act(argparse.Action):
        def __call__(self, parser, namespace, fname, option_string=None):
            ext = os.path.splitext(fname[0])[1][1:]
            if ext not in choices:
                option_string = '({})'.format( option_string ) if option_string else ''
                parser.error( "file doesn't end with {}{}".format( choices, option_string ) )
            else:
                setattr(namespace, self.dest, fname)
    return Act

# ------------------------------------------------------------------------------
# command line arguments
parser = argparse.ArgumentParser()

group_test = parser.add_argument_group( 'test' )
group_test.add_argument( '-t', '--test', action='store',
    help='test command to run, e.g., --test "mpirun -np 4 ./test"; default "%(default)s"',
    default='./test' )
group_test.add_argument( '--xml', action=check_file_ext( {'xml'} ),
    help='generate report.xml for jenkins',
    nargs=1 )

group_size = parser.add_argument_group( 'matrix dimensions (default is medium)' )
group_size.add_argument( '-x', '--xsmall', action='store_true', help='run x-small tests' )
group_size.add_argument( '-s', '--small',  action='store_true', help='run small tests' )
group_size.add_argument( '-m', '--medium', action='store_true', help='run medium tests' )
group_size.add_argument( '-l', '--large',  action='store_true', help='run large tests' )
group_size.add_argument(       '--square', action='store_true', help='run square (m = n = k) tests', default=False )
group_size.add_argument(       '--tall',   action='store_true', help='run tall (m > n) tests', default=False )
group_size.add_argument(       '--wide',   action='store_true', help='run wide (m < n) tests', default=False )
group_size.add_argument(       '--mnk',    action='store_true', help='run tests with m, n, k all different', default=False )
group_size.add_argument(       '--dim',    action='store',      help='explicitly specify size', default='' )

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
group_opt.add_argument( '--alpha',  action='store', help='default=%(default)s', default='' )
group_opt.add_argument( '--beta',   action='store', help='default=%(default)s', default='' )
group_opt.add_argument( '--incx',   action='store', help='default=%(default)s', default='1,2,-1,-2' )
group_opt.add_argument( '--incy',   action='store', help='default=%(default)s', default='1,2,-1,-2' )
group_opt.add_argument( '--align',  action='store', help='default=%(default)s', default='32' )
group_opt.add_argument( '--check',  action='store', help='default=y', default='' )  # default in test.cc
group_opt.add_argument( '--ref',    action='store', help='default=y', default='' )  # default in test.cc

# LAPACK only
group_opt.add_argument( '--itype',  action='store', help='default=%(default)s', default='1,2,3' )
group_opt.add_argument( '--factored', action='store', help='default=%(default)s', default='f,n,e' )
group_opt.add_argument( '--equed',  action='store', help='default=%(default)s', default='n,r,c,b' )
group_opt.add_argument( '--direct', action='store', help='default=%(default)s', default='f,b' )
group_opt.add_argument( '--storev', action='store', help='default=%(default)s', default='c,r' )
group_opt.add_argument( '--norm',   action='store', help='default=%(default)s', default='max,1,inf,fro' )
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
group_opt.add_argument( '--kd',     action='store', help='default=%(default)s', default='20,100' )
group_opt.add_argument( '--kl',     action='store', help='default=%(default)s', default='20,100' )
group_opt.add_argument( '--ku',     action='store', help='default=%(default)s', default='20,100' )
group_opt.add_argument( '--vl',     action='store', help='default=%(default)s', default='-inf,0' )
group_opt.add_argument( '--vu',     action='store', help='default=%(default)s', default='inf' )
group_opt.add_argument( '--il',     action='store', help='default=%(default)s', default='10' )
group_opt.add_argument( '--iu',     action='store', help='default=%(default)s', default='-1,100' )
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
if (not (opts.xsmall or opts.small or opts.medium or opts.large)):
    opts.medium = True

# by default, run all shapes
if (not (opts.square or opts.tall or opts.wide or opts.mnk)):
    opts.square = True
    opts.tall   = True
    opts.wide   = True
    opts.mnk    = True

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

# LAPACK only
itype  = ' --itype '  + opts.itype  if (opts.itype)  else ''
factored = ' --factored ' + opts.factored if (opts.factored)  else ''
equed  = ' --equed '  + opts.equed  if (opts.equed)  else ''
direct = ' --direct ' + opts.direct if (opts.direct) else ''
storev = ' --storev ' + opts.storev if (opts.storev) else ''
norm   = ' --norm '   + opts.norm   if (opts.norm)   else ''
jobz   = ' --jobz '   + opts.jobz   if (opts.jobz)   else ''
jobu   = ' --jobu '   + opts.jobu   if (opts.jobu)   else ''
jobvt  = ' --jobvt '  + opts.jobvt  if (opts.jobvt)  else ''
jobvl  = ' --jobvl '  + opts.jobvl  if (opts.jobvl)  else ''
jobvr  = ' --jobvr '  + opts.jobvr  if (opts.jobvr)  else ''
jobvs  = ' --jobvs '  + opts.jobvs  if (opts.jobvs)  else ''
balanc = ' --balanc ' + opts.balanc if (opts.balanc)   else ''
sort   = ' --sort '   + opts.sort   if (opts.sort)   else ''
sense  = ' --sense '  + opts.sense  if (opts.sense)   else ''
vect   = ' --vect '   + opts.vect   if (opts.vect)   else ''
l      = ' --l '      + opts.l      if (opts.l)      else ''
kd     = ' --kd '     + opts.kd     if (opts.kd)     else ''
kl     = ' --kl '     + opts.kl     if (opts.kl)     else ''
ku     = ' --ku '     + opts.ku     if (opts.ku)     else ''
vl     = ' --vl '     + opts.vl     if (opts.vl)     else ''
vu     = ' --vu '     + opts.vu     if (opts.vu)     else ''
il     = ' --il '     + opts.il     if (opts.il)     else ''
iu     = ' --iu '     + opts.iu     if (opts.iu)     else ''
mtype  = ' --matrixtype ' + opts.matrixtype if (opts.matrixtype) else ''

# general options for all routines
gen = check + ref

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
    [ 'gesv',  gen + dtype + align + n ],
    [ 'gesvx', gen + dtype + align + n + factored + trans + equed ],
    [ 'getrf', gen + dtype + align + mn ],
    [ 'getrs', gen + dtype + align + n + trans ],
    [ 'getri', gen + dtype + align + n ],
    [ 'gecon', gen + dtype + align + n ],
    [ 'gerfs', gen + dtype + align + n + trans ],
    [ 'geequ', gen + dtype + align + n ],
    ]

# General Banded
if (opts.gb):
    cmds += [
    [ 'gbsv',  gen + dtype + align + n  + kl + ku ],
    [ 'gbtrf', gen + dtype + align + mn + kl + ku ],
    [ 'gbtrs', gen + dtype + align + n  + kl + ku + trans ],
    [ 'gbcon', gen + dtype + align + n  + kl + ku ],
    [ 'gbrfs', gen + dtype + align + n  + kl + ku + trans ],
    [ 'gbequ', gen + dtype + align + n  + kl + ku ],
    ]

# General Tri-Diagonal
if (opts.gt):
    cmds += [
    [ 'gtsv',  gen + dtype + align + n ],
    [ 'gttrf', gen + dtype +         n ],
    [ 'gttrs', gen + dtype + align + n + trans ],
    [ 'gtcon', gen + dtype +         n ],
    [ 'gtrfs', gen + dtype + align + n + trans ],
    ]

# Cholesky
if (opts.chol):
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

# symmetric indefinite, Bunch-Kaufman
if (opts.sysv):
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
    [ 'spcon', gen + dtype +         n + uplo ],
    [ 'sprfs', gen + dtype + align + n + uplo ],
    ]

# symmetric indefinite, rook
if (opts.rook):
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
if (opts.aasen):
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
if (opts.hesv):
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
if (opts.least_squares):
    cmds += [
    [ 'gels',   gen + dtype + align + mn + trans_nc ],
    [ 'gelsy',  gen + dtype + align + mn ],
    [ 'gelsd',  gen + dtype + align + mn ],
    [ 'gelss',  gen + dtype + align + mn ],
    [ 'getsls', gen + dtype + align + mn + trans_nc ],

    # Generalized
    [ 'gglse', gen + dtype + align + mnk ],
    [ 'ggglm', gen + dtype + align + mnk ],
    ]

# QR
if (opts.qr):
    cmds += [
    [ 'geqrf', gen + dtype + align + n + wide + tall ],
    [ 'ggqrf', gen + dtype + align + mnk ],
    [ 'ungqr', gen + dtype + align + mn ],  # m >= n
    #[ 'unmqr', gen + dtype_real    + align + mnk + side + trans    ],  # real does trans = N, T, C
    #[ 'unmqr', gen + dtype_complex + align + mnk + side + trans_nc ],  # complex does trans = N, C, not T

    # Triangle-pentagon
    [ 'tpqrt',  gen + dtype + align + mn + l ],
    [ 'tpqrt2', gen + dtype + align + mn + l ],
    [ 'tpmqrt', gen + dtype_real    + align + mn + l + side + trans    ],  # real does trans = N, T, C
    [ 'tpmqrt', gen + dtype_complex + align + mn + l + side + trans_nc ],  # complex does trans = N, C, not T
    #[ 'tprfb',  gen + dtype + align + mn + l ],  # TODO: bug in LAPACKE crashes tester
    ]

# LQ
if (opts.lq):
    cmds += [
    [ 'gelqf', gen + dtype + align + mn ],
    #[ 'gglqf', gen + dtype + align + mn ],
    [ 'unglq', gen + dtype + align + mn ],  # m <= n, k <= m  TODO Fix the input sizes to match constraints
    #[ 'unmlq', gen + dtype_real    + align + mnk + side + trans    ],  # real does trans = N, T, C
    #[ 'unmlq', gen + dtype_complex + align + mnk + side + trans_nc ],  # complex does trans = N, C, not T

    # Triangle-pentagon
    [ 'tplqt',  gen + dtype + align + mn + l ],
    [ 'tplqt2', gen + dtype + align + mn + l ],
    [ 'tpmlqt', gen + dtype_real    + align + mn + l + side + trans    ],  # real does trans = N, T, C
    [ 'tpmlqt', gen + dtype_complex + align + mn + l + side + trans_nc ],  # complex does trans = N, C, not T
    ]

# QL
if (opts.ql):
    cmds += [
    [ 'geqlf', gen + dtype + align + mn ],
    #[ 'ggqlf', gen + dtype + align + mn ],
    [ 'ungql', gen + dtype + align + mn ],
    #[ 'unmql', gen + dtype_real    + align + mnk + side + trans    ],  # real does trans = N, T, C
    #[ 'unmql', gen + dtype_complex + align + mnk + side + trans_nc ],  # complex does trans = N, C, not T
    ]

# RQ
if (opts.rq):
    cmds += [
    [ 'gerqf', gen + dtype + align + mn ],
    [ 'ggrqf', gen + dtype + align + mnk ],
    [ 'ungrq', gen + dtype + align + mnk ],
    #[ 'unmrq', gen + dtype_real    + align + mnk + side + trans    ],  # real does trans = N, T, C
    #[ 'unmrq', gen + dtype_complex + align + mnk + side + trans_nc ],  # complex does trans = N, C, not T
    ]

# symmetric eigenvalues
if (opts.syev):
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
if (opts.sygv):
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
    [ 'hbgv',  gen + dtype + align + n + jobz + uplo + kd ],
    [ 'hbgvx', gen + dtype + align + n + jobz + uplo + kd + vl + vu ],
    [ 'hbgvx', gen + dtype + align + n + jobz + uplo + kd + il + iu ],
    #[ 'hbgvd',  gen + dtype + align + n + jobz + uplo + kd ],
    #[ 'hbgvr', gen + dtype + align + n + uplo ],
    [ 'hbgst', gen + dtype + align + n + vect + uplo + kd ],
    ]

# non-symmetric eigenvalues
if (opts.geev):
    cmds += [
    [ 'geev',  gen + dtype + align + n + jobvl + jobvr ],
    [ 'ggev',  gen + dtype + align + n + jobvl + jobvr ],
    #[ 'geevx', gen + dtype + align + n + balanc + jobvl + jobvr + sense ],
    [ 'gehrd', gen + dtype + align + n ],
    [ 'unghr', gen + dtype + align + n ],
    [ 'unmhr', gen + dtype_real    + align + mn + side + trans    ],  # real does trans = N, T, C
    [ 'unmhr', gen + dtype_complex + align + mn + side + trans_nc ],  # complex does trans = N, C, not T
    #[ 'trevc', gen + dtype + align + n + side + howmany + select ],
    #[ 'geesx', gen + dtype + align + n + jobvs + sort + select + sense ],
    ]

# svd
if (opts.svd):
    cmds += [
    [ 'gesvd',         gen + dtype + align + mn + jobu + jobvt ],
    [ 'gesdd',         gen + dtype + align + mn + jobu ],
    [ 'gesvdx',        gen + dtype + align + mn + jobz + jobvr + vl + vu ],
    [ 'gesvdx',        gen + dtype + align + mn + jobz + jobvr + il + iu ],
    #[ 'gesvd_2stage',  gen + dtype + align + mn ],
    #[ 'gesdd_2stage',  gen + dtype + align + mn ],
    #[ 'gesvdx_2stage', gen + dtype + align + mn ],
    #[ 'gejsv',         gen + dtype + align + mn ],
    #[ 'gesvj',         gen + dtype + align + mn + joba + jobu + jobv ],
    ]

# auxilary
if (opts.aux):
    cmds += [
    [ 'lacpy', gen + dtype + align + mn + mtype ],
    [ 'laset', gen + dtype + align + mn + mtype ],
    [ 'laswp', gen + dtype + align + mn ],
    ]

# auxilary - householder
if (opts.aux_house):
    cmds += [
    [ 'larfg', dtype         + n   + incx_pos ],
    [ 'larf',  gen + dtype + align + mn  + incx + side ],
    [ 'larfx', gen + dtype + align + mn  + side ],
    [ 'larfb', gen + dtype + align + mnk + side + trans + direct + storev ],
    [ 'larft', gen + dtype + align + nk  + direct + storev ],
    ]

# auxilary - norms
if (opts.aux_norm):
    cmds += [
    [ 'lange', gen + dtype + align + mn + norm ],
    [ 'lanhe', gen + dtype + align + n  + norm + uplo ],
    [ 'lansy', gen + dtype + align + n  + norm + uplo ],
    [ 'lantr', gen + dtype + align + mn + norm + uplo + diag ],

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
    ]

# additional blas
if (opts.blas):
    cmds += [
    [ 'syr',   gen + dtype + align + n + uplo ],
    ]

# ------------------------------------------------------------------------------
# when output is redirected to file instead of TTY console,
# print extra messages to stderr on TTY console.
output_redirected = not sys.stdout.isatty()

# ------------------------------------------------------------------------------
# cmd is a pair of strings: (function, args)

def run_test( cmd ):
    cmd = opts.test +' '+ cmd[0] +' '+ cmd[1]
    print( cmd, file=sys.stderr )
    output = ''
    p = subprocess.Popen( cmd.split(), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT )
    # Read unbuffered ("for line in p.stdout" will buffer).
    for line in iter(p.stdout.readline, b''):
        print( line, end='' )
        output += line
    err = p.wait()
    if (err < 0):
        print( 'FAILED: exit with signal', -err )
    return (err, output)
# end

# ------------------------------------------------------------------------------
failed_tests = []
passed_tests = []
ntests = len(opts.tests)
run_all = (ntests == 0)

for cmd in cmds:
    if (run_all or cmd[0] in opts.tests):
        if (not run_all):
            opts.tests.remove( cmd[0] )
        (err, output) = run_test( cmd )
        if (err):
            failed_tests.append( (cmd[0], err, output) )
        else:
            passed_tests.append( cmd[0] )
if (opts.tests):
    print( 'Warning: unknown routines:', ' '.join( opts.tests ))

# print summary of failures
nfailed = len( failed_tests )
if (nfailed > 0):
    print( '\n' + str(nfailed) + ' routines FAILED:',
           ', '.join( [x[0] for x in failed_tests] ),
           file=sys.stderr )

# generate jUnit compatible test report
if opts.xml:
    report_file_name = opts.xml[0]
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
    tree.write(report_file_name)
# end

exit( nfailed )
