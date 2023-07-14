#!/usr/bin/env python3

'''
wrapper_gen.py generates C++ wrappers, header prototypes, and testers for the
LAPACK Fortran routines by parsing doxygen comments in LAPACK's SRC *.f files.
It needs a copy of Netlib LAPACK source code, pointed to by $LAPACKDIR.
It takes a list of routines, without precision prefix (so "getrf" instead of "sgetrf", "dgetrf", ...).
It generates files in lapackpp/gen/ directory, which can then be manually moved to the appropriate place.

Arguments:
    -H, --header   generate header   in gen/wrappers.hh to go in include/lapack/wrappers.hh
    -w, --wrapper  generate wrappers in gen/foo.cc      to go in src
    -t, --tester   generate testers  in gen/test_foo.cc to go in test

Example creating tester:

    # Assumes $LAPACKDIR is set
    lapackpp> echo $LAPACKDIR
    /Users/mgates/Documents/lapack

    # Generate Fortran prototypes. Need to reformat nicely.
    lapackpp> ./tools/header_gen.py {s,d,c,z}posv
    generating gen/fortran.h

    # If necessary, add the routine to tools/first_version.py,
    # which is first version of LAPACK to include the routine.
    lapackpp> edit test/first_version.py

    # Generate LAPACK++ wrapper and tester.
    lapackpp> ./tools/wrapper_gen.py posv
    generating gen/wrappers.hh
    generating gen/posv.cc
    generating gen/test_posv.cc
        /Users/mgates/Documents/lapack/SRC/sposv.f
        /Users/mgates/Documents/lapack/SRC/dposv.f
        /Users/mgates/Documents/lapack/SRC/cposv.f
        /Users/mgates/Documents/lapack/SRC/zposv.f

    lapackpp> mv gen/test_posv.cc test/
    lapackpp> mv gen/posv.cc      src/

    lapackpp> edit include/lapack/wrappers.hh  # add gen/wrappers.hh
    lapackpp> edit CMakeLists.txt      # add posv.cc line
    lapackpp> edit test/CMakeLists.txt # add test_posv.cc line
    lapackpp> edit test/test.cc        # add posv line
    lapackpp> edit test/test.hh        # add posv line
    lapackpp> edit test/run_tests.sh   # add posv line

    lapackpp> make

    # Test. Initial version fails because matrix isn't positive definite.
    lapackpp> ./test/run_tests.py posv
    input: ./test posv
                                               LAPACK++     LAPACK++     LAPACK++         Ref.         Ref.
      type    uplo       n    nrhs   align        error     time (s)      Gflop/s     time (s)      Gflop/s  status
    lapack::posv returned error 2
    LAPACKE_posv returned error 2
         d   lower     100      10       1   0.0000e+00       0.0000      20.7156       0.0000      14.2013  pass
    ...
    All tests passed.

    # Resolve tests. In this case, add code to make matrix positive
    # definite (e.g., diagonally dominant)
    lapackpp> edit test_posv.cc
    lapackpp> make

    # now tests pass; commit changes.
    lapackpp/test> ./test posv --type s,d,c,z --dim 100:300:100
    input: ./test posv --type s,d,c,z --dim 100:300:100
                                               LAPACK++     LAPACK++     LAPACK++         Ref.         Ref.
      type    uplo       n    nrhs   align        error     time (s)      Gflop/s     time (s)      Gflop/s  status
         s   lower     100      10       1   0.0000e+00       0.0002       2.6913       0.0001       4.2364  pass
         s   lower     200      10       1   0.0000e+00       0.0005       6.4226       0.0004       7.9437  pass
         s   lower     300      10       1   0.0000e+00       0.0008      14.2683       0.0008      14.0091  pass
    ...
    All tests passed.

    # Add new files, see what changed, and commit it.
    lapackpp> git add src/posv.cc test/test_posv.cc
    lapackpp> git diff
    lapackpp> git commit -m 'add posv' .
'''

from __future__ import print_function

import sys
import re
import os
import argparse
import traceback

from text_balanced import extract_bracketed
from first_version import first_version

parser = argparse.ArgumentParser()
parser.add_argument( '-H', '--header',  action='store_true',
                     help='generate header   in gen/wrappers.hh to go in include/lapack/wrappers.hh' )

parser.add_argument( '-w', '--wrapper', action='store_true',
                     help='generate wrappers in gen/foo.cc      to go in src' )

parser.add_argument( '-t', '--tester',  action='store_true',
                     help='generate testers  in gen/test_foo.cc to go in test' )

parser.add_argument( '-d', '--debug',   action='store_true',
                     help='debug mode' )

parser.add_argument( 'argv', nargs=argparse.REMAINDER )
args = parser.parse_args()

if (len( args.argv ) == 0):
    parser.print_help()
    print( '\nBy default does all three (-H -w -t). See script for more details' )
    exit(1)
# end

# default: do all
if (not (args.header or args.tester or args.wrapper)):
    args.header  = True
    args.tester  = True
    args.wrapper = True

debug = args.debug

gen = 'gen'
if (not os.path.exists( gen )):
    os.mkdir( gen )

# ------------------------------------------------------------------------------
# configuration

# --------------------
# for wrappers.hh
header_top = '''\
#ifndef LAPACK_WRAPPERS_HH
#define LAPACK_WRAPPERS_HH

#include "lapack/util.hh"

namespace lapack {

'''

header_bottom = '''\

}  // namespace lapack

#endif // LAPACK_WRAPPERS_HH
'''

# --------------------
# for src/*.cc wrappers
wrapper_top1 = '''\
// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"

'''

wrapper_top2 = '''\
#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

'''

wrapper_bottom = '''\
}  // namespace lapack
'''

# --------------------
# for test/test_*.cc testers
tester_top = '''\
// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "lapack.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

'''

tester_bottom = ''

# --------------------
# 4 space indent
tab = '    '

# ------------------------------------------------------------------------------
typemap = {
    'character'        : 'char',
    'integer'          : 'int64_t',
    'real'             : 'float',
    'double precision' : 'double',
    'complex'          : 'std::complex<float>',
    'complex*16'       : 'std::complex<double>',
    'logical'          : 'bool',

    'logical function of two real arguments'               : 'lapack_s_select2',
    'logical function of three real arguments'             : 'lapack_s_select3',

    'logical function of two double precision arguments'   : 'lapack_d_select2',
    'logical function of three double precision arguments' : 'lapack_d_select3',

    'logical function of one complex argument'     : 'lapack_c_select1',
    'logical function of two complex arguments'    : 'lapack_c_select2',

    'logical function of one complex*16 argument'  : 'lapack_z_select1',
    'logical function of two complex*16 arguments' : 'lapack_z_select2',
}

# ------------------------------------------------------------------------------
# enum mappings
# each tuple is (enum, enum2char, map), where
# enum2char is function to convert enum to LAPACK char,
# map goes from LAPACK char to enum values

# --------------------
# see blas_util.hh
uplo = ('lapack::Uplo', 'uplo2char', {
    'u': 'Upper',
    'l': 'Lower',
    'g': 'General',
})

diag = ('lapack::Diag', 'diag2char', {
    'n': 'NonUnit',
    'u': 'Unit',
})

op = ('lapack::Op', 'op2char', {
    'n': 'NoTrans',
    't': 'Trans',
    'c': 'ConjTrans',
})

side = ('lapack::Side', 'side2char', {
    'l': 'Left',
    'r': 'Right',
})

# --------------------
# see lapack_util.hh
sides = ('lapack::Sides', 'sides2char', {
    'l': 'Left',
    'r': 'Right',
    'b': 'Both',
})

norm = ('lapack::Norm', 'norm2char', {
    '1': 'One',
    'o': 'One',
    'i': 'Inf',
    'f': 'Fro',
    'm': 'Max',
})

job = ('lapack::Job', 'job2char', {
    'n': 'NoVec',
    'v': 'Vec',
    'u': 'UpdateVec',

    'a': 'AllVec',
    's': 'SomeVec',
    'o': 'OverwriteVec',

    'p': 'CompactVec',
    'c': 'SomeVecTol',
    'j': 'AllVecJacobi',
    'w': 'Workspace',
})

# ----- custom job maps
#job_bdsdc = ('lapack::Job', 'job_bdsdc2char', {
#    'n': 'NoVec',
#    'i': 'Vec',
#    'p': 'CompactVec',
#})

job_csd = ('lapack::Job', 'job_csd2char', {
    'n': 'NoVec',
    'y': 'UpdateVec',
})

# hseqr, tgsja, trsen, gghd3, etc.
job_comp = ('lapack::Job', 'job_comp2char', {
    'n': 'NoVec',
    'i': 'Vec',
    'v': 'UpdateVec',
    'p': 'CompactVec',  # bdsdc
})

# tgsja
job_compu = ('lapack::Job', 'job_compu2char', {
    'n': 'NoVec',
    'i': 'Vec',
    'u': 'UpdateVec',
})

# tgsja
job_compq = ('lapack::Job', 'job_compq2char', {
    'n': 'NoVec',
    'i': 'Vec',
    'q': 'UpdateVec',
})

# ggsvd, ggsvp3
jobu = ('lapack::Job', 'jobu2char', {
    'n': 'NoVec',
    'u': 'Vec',
})

# ggsvd, ggsvp3
jobq = ('lapack::Job', 'jobq2char', {
    'n': 'NoVec',
    'q': 'Vec',
})

jobu_gejsv = ('lapack::Job', 'job_gejsv2char', {
    'n': 'NoVec',
    'u': 'SomeVec',
    'f': 'AllVec',
    'w': 'Work',
})

job_gesvj = ('lapack::Job', 'job_gesvj2char', {
    'n': 'NoVec',
    'u': 'SomeVec',     # jobu
    'c': 'SomeVecTol',  # jobu
    'v': 'Vec',         # jobv
    'a': 'UpdateVec',   # jobv
})
# -----

# hseqr
jobschur = ('lapack::JobSchur', 'jobschur2char', {
    'e': 'Eigenvalues',
    's': 'Schur',
})

# gees
sort = ('lapack::Sort', 'sort2char', {
    'n': 'NotSorted',
    's': 'Sorted',
})

# syevx, geevx, gesvdx
range_enum = ('lapack::Range', 'range2char', {
    'a': 'All',
    'v': 'Value',
    'i': 'Index',
})

# ormbr, orgbr
vect = ('lapack::Vect', 'vect2char', {
    'q': 'Q',
    'p': 'P',
    'n': 'None',
    'b': 'Both',
})

# larfb
direction = ('lapack::Direction', 'direction2char', {
    'f': 'Forward',
    'b': 'Backward',
})

# larfb
storev = ('lapack::StoreV', 'storev2char', {
    'c': 'Columnwise',
    'r': 'Rowwise',
})

# lascl, laset
matrixtype = ('lapack::MatrixType', 'matrixtype2char', {
    'g': 'General',
    'l': 'Lower',
    'u': 'Upper',
    'h': 'Hessenberg',
    'b': 'LowerBand',
    'q': 'UpperBand',
    'z': 'Band',
})

# trevc
howmany = ('lapack::HowMany', 'howmany2char', {
    'a': 'All',
    'b': 'Backtransform',
    's': 'Select',
})

# *svx, *rfsx
equed = ('lapack::Equed', 'equed2char', {
    'n': 'None',
    'r': 'Row',
    'c': 'Col',
    'b': 'Both',
    'y': 'Yes',  # porfsx
})

# *svx, *rfsx
factored = ('lapack::Factored', 'factored2char', {
    'f': 'Factored',
    'n': 'NotFactored',
    'e': 'Equilibrate',
})

# trsen, geesx
sense = ('lapack::Sense', 'sense2char', {
    'n': 'None',
    'e': 'Eigenvalues',
    'v': 'Subspace',
    'b': 'Both',
})

# disna
jobcond = ('lapack::JobCond', 'jobcond2char', {
    'e': 'EigenVec',
    'l': 'LeftSingularVec',
    'r': 'RightSingularVec',
})

# gebak, gebal
balance = ('lapack::Balance', 'balance2char', {
    'n': 'None',
    'p': 'Permute',
    's': 'Scale',
    'b': 'Both',
})

# stebz, larrd
order = ('lapack::Order', 'order2char', {
    'b': 'Block',
    'e': 'Entire',
})

# rowcol
rowcol = ('lapack::RowCol', 'rowcol2char', {
    'c': 'Col',
    'r': 'Row',
})

# --------------------
# maps argument names to enum tuples
enum_map = {
    # blas
    'uplo':         uplo,
    'diag':         diag,
    'trans':        op,
    'transr':       op,         # hfrk
    'side':         side,

    # lapack
    'sides':        sides,      # trevc
    'norm':         norm,

    # ----- jobs
    'compq':        job_comp,   # bdsdc, gghrd, hgeqz, trexc, trsen
    'compz':        job_comp,   # gghrd, hgeqz, hseqr, pteqr, stedc, steqr

    'jobq':         jobq,       # ggsvd3, ggsvp3
    'jobq_tgsja':   job_compq,  # tgsja

    'jobu':         job,        # gesvd, gesvdx
    'jobu_gejsv':   jobu_gejsv, # gejsv
    'jobu_gesvj':   job_gesvj,  # gesvj
    'jobu_ggsvd':   jobu,       # ggsvd3, ggsvp3
    'jobu_tgsja':   job_compu,  # tgsja

    'jobu1':        job_csd,    # bbcsd, orcsd2by1
    'jobu2':        job_csd,    # bbcsd, orcsd2by1
    'jobv1t':       job_csd,    # bbcsd, orcsd2by1
    'jobv2t':       job_csd,    # bbcsd, orcsd2by1

    'jobv':         job,        # gejsv, ggsvd3, ggsvp3
    'jobv_gesvj':   job_gesvj,  # gesvj
    'jobv_tgsja':   job_comp,   # tgsja

    'jobvl':        job,        # geev, ggev[3]
    'jobvr':        job,        # geev, ggev[3]
    'jobvs':        job,        # gees[x]
    'jobvsl':       job,        # gges[3x]
    'jobvsr':       job,        # gges[3x]
    'jobvt':        job,        # gesvd, gesvdx
    'jobz':         job,        # bdsvdx, gesdd, {hb,he,hp,st}{ev,gv}*, stegr, stemr, [hbgst, hbtrd, hetrd_2stage: was vect]

    'jobschur':     jobschur,   # hseqr
    'sort':         sort,       # gees
    'range':        range_enum, # syevx, geevx, gesvdx
    'vect':         vect,       # ormbr, orgbr
    'direction':    direction,  # larfb
    'storev':       storev,     # larfb
    'matrixtype':   matrixtype, # lascl, laset
    'howmany':      howmany,    # gehrd
    'equed':        equed,      # *svx, *rfsx
    'fact':         factored,   # *svx, *rfsx
    'sense':        sense,      # trsen, geesx
    'jobcond':      jobcond,    # disna
    'balance':      balance,    # {ge,gg}{bak,bal}
    'order':        order,      # stebz, larrd
}

# ------------------------------------------------------------------------------
# maps function to list of (search, replace) pairs
arg_rename = {
    # function         search    replace
    'gebak':        (( 'JOB',    'BALANCE'      ), ),
    'ggbak':        (( 'JOB',    'BALANCE'      ), ),
    'gebal':        (( 'JOB',    'BALANCE'      ), ),
    'ggbal':        (( 'JOB',    'BALANCE'      ), ),

    'disna':        (( 'JOB',    'JOBCOND'      ), ),

    'hetrd_2stage': (( 'VECT',   'JOBZ'         ), ),
    'sytrd_2stage': (( 'VECT',   'JOBZ'         ), ),
    'hbgst':        (( 'VECT',   'JOBZ'         ), ),
    'sbgst':        (( 'VECT',   'JOBZ'         ), ),
    'hbtrd':        (( 'VECT',   'JOBZ'         ), ),
    'sbtrd':        (( 'VECT',   'JOBZ'         ), ),

    'tgsja':        (( 'JOBU',   'JOBU_TGSJA'   ),
                     ( 'JOBV',   'JOBV_TGSJA'   ),
                     ( 'JOBQ',   'JOBQ_TGSJA'   ), ),

    'gejsv':        (( 'JOBU',   'JOBU_GEJSV'   ),
                     ( 'JOBV',   'JOBV_GEJSV'   ), ),

    'gesvj':        (( 'JOBU',   'JOBU_GESVJ'   ),
                     ( 'JOBV',   'JOBV_GESVJ'   ),
                     ( 'JOBA',   'UPLO'         ), ),

    'ggsvd':        (( 'JOBU',   'JOBU_GGSVD'   ), ),
    'ggsvd3':       (( 'JOBU',   'JOBU_GGSVD'   ), ),
    'ggsvp':        (( 'JOBU',   'JOBU_GGSVD'   ), ),
    'ggsvp3':       (( 'JOBU',   'JOBU_GGSVD'   ), ),

    'trevc':        (( 'HOWMNY', 'HOWMANY'      ),
                     ( 'SIDE',   'SIDES'        ), ),

    'trevc3':       (( 'HOWMNY', 'HOWMANY'      ),
                     ( 'SIDE',   'SIDES'        ), ),

    'hseqr':        (( 'JOB',    'JOBSCHUR'     ), ),
    'hgeqz':        (( 'JOB',    'JOBSCHUR'     ), ),

    'trsen':        (( 'JOB',    'SENSE'        ), ),  # match geesx

    'bdsvdx':       (( 'NS',     'NFOUND'       ), ),
    'gesvdx':       (( 'NS',     'NFOUND'       ), ),
    'heevr':        (( 'M',      'NFOUND'       ), ),
    'syevr':        (( 'M',      'NFOUND'       ), ),
    'heevr_2stage': (( 'M',      'NFOUND'       ), ),
    'syevr_2stage': (( 'M',      'NFOUND'       ), ),
    'heevx':        (( 'M',      'NFOUND'       ), ),
    'syevx':        (( 'M',      'NFOUND'       ), ),
    'heevx_2stage': (( 'M',      'NFOUND'       ), ),
    'syevx_2stage': (( 'M',      'NFOUND'       ), ),

    'lanhb':        (( 'K',      'KD'           ), ),
    'lansb':        (( 'K',      'KD'           ), ),

    'lascl':        (( 'TYPE',   'MATRIXTYPE'   ), ),
}

# ------------------------------------------------------------------------------
# replacements applied as last step
# these reverse some temporary changes done by arg_rename
post_rename = {
    # function         replace       search
    'tgsja':        (( 'jobu_tgsja', 'jobu' ),
                     ( 'jobv_tgsja', 'jobv' ),
                     ( 'jobq_tgsja', 'jobq' ), ),

    'gejsv':        (( 'jobu_gejsv', 'jobu' ),
                     ( 'jobv_gejsv', 'jobv' ), ),

    'gesvj':        (( 'jobu_gesvj', 'jobu' ),
                     ( 'jobv_gesvj', 'jobv' ), ),

    'ggsvd':        (( 'jobu_ggsvd', 'jobu' ), ),
    'ggsvd3':       (( 'jobu_ggsvd', 'jobu' ), ),
    'ggsvp':        (( 'jobu_ggsvd', 'jobu' ), ),
    'ggsvp3':       (( 'jobu_ggsvd', 'jobu' ), ),
}

# ------------------------------------------------------------------------------
alias_map = {
    'sy': 'he',
    'sp': 'hp',
    'sb': 'hb',
    'or': 'un',
    'op': 'up',
}

# ------------------------------------------------------------------------------
fortran_operators = {
    'LT': '<',
    'lt': '<',

    'GT': '>',
    'gt': '>',

    'LE': '<=',
    'le': '<=',

    'GE': '>=',
    'ge': '>=',

    'EQ': '==',
    'eq': '==',

    'NE': '!=',
    'ne': '!=',
}

# ------------------------------------------------------------------------------
# list of LAPACK functions that begin with precision [sdcz].
# excludes i and mixed precision prefixes (dsposv, scsum1, etc.)

(top, script) = os.path.split( sys.argv[0] )
f = open( os.path.join( top, 'functions.txt' ))
lapack_functions = []
for line in f:
    lapack_functions.append( line.rstrip() )
lapack_functions_re = re.compile( r'\b[SDCZ](' + '|'.join( lapack_functions ).upper() + r')\b' )

# ------------------------------------------------------------------------------
# captures information about each function argument
class Arg:
    def __init__( self, name, intent ):
        self.name      = name
        self.name_orig = name
        self.intent    = intent
        self.desc      = ''

        self.use_query = False
        self.is_enum   = False
        self.is_array  = False
        self.dtype     = None
        self.dim       = None
        self.lbound    = None
        self.is_work   = False
        self.is_lwork  = False
    # end
# end

# ------------------------------------------------------------------------------
# captures information about each function
class Func:
    def __init__( self, xname ):
        self.xname   = xname
        self.is_func = False  # Fortran function or subroutine (default)
        self.args    = []
        self.retval  = 'void'
        self.purpose = ''
        self.details = ''
        self.group   = 'unknown'

        # get base name of function (without precision)
        s = (re.search( r'^(?:ds|zc)(gesv|posv)', xname ) or
             re.search( r'^[sdcz](\w+)', xname ))
        if (s):
            self.name = s.group(1)
        else:
            self.name = 'unknown'
            print( 'unknown base function name', xname )
    # end
# end

# ------------------------------------------------------------------------------
# joins a and b with ";"
# if a is empty, returns b. Useful for building up lists:
#     foo = ''
#     foo = join( foo, 'arg i' )
def join( a, b ):
    if (a):
        return a + '; ' + b
    else:
        return b
# end

# ------------------------------------------------------------------------------
# Parses an arg's description for a dimension statement.
# Tries to find all dimensions that exist.
# Sets arg.dim to string of all dimensions, seperated by ";".
def parse_dim( arg ):
    if (debug): print( 'parse_dim' )
    txt = arg.desc

    # easy case: dimension at end of line
    s = re.search( r'array,\s+(dimensions?|length|size)\s+(\w+|\(.*?\))\.?\n', txt )
    if (s):
        arg.dim = s.group(2).lower()
        if (debug > 1): print( ':: easy = ' + arg.dim )
        return

    # harder cases
    # find initial dimensions keyword, and remainder after that
    state = 0
    s = re.search( r'array,\s+(dimensions?|length|size)\s*(.*)', txt, re.DOTALL )
    if (not s):
        if (debug > 1): print( ':: error: no match' )
        arg.dim = 'ERROR PARSING (1)'
        return

    arg.dim = ''
    rem = s.group(2)

    length = len(rem)
    while (length > 0):
        s = re.search( r'^(?:or|and)\s+(.*)', rem, re.DOTALL )
        if (s):
            if (debug > 1): print( ':: text = or/and' )
            rem = s.group(1)

        s = re.search( r'^at least\s*(.*)', rem, re.DOTALL )
        if (s):
            if (debug > 1): print( ':: text = at least' )
            rem = s.group(1)

        if (rem[0] == '('):
            # (paren) expression like (MAX(1,N))
            (d, rem) = extract_bracketed( rem, '(', ')' )
            if (d):
                if (debug > 1): print( ':: expr =', d )
                arg.dim = join( arg.dim, d.lower() )
                rem = rem.lstrip(' ')
            else:
                if (debug > 1): print( ':: error: bad expr' )
                arg.dim += '; ERROR PARSING (2)'
                break
        else:
            if (debug > 1): print( ':: missing expr -- implicit end' )
            break
        # end

        # multiple dimensions
        s = re.search( r'^((?:if|when|otherwise)[^\n]*)\n *(.*)', rem, re.DOTALL )
        if (s):
            if (debug > 1): print( ':: cond =', s.group(1) )
            arg.dim += ' ' + s.group(1)
            rem = s.group(2)

        # end of line
        if (re.search( r'^\.?\n', rem )):
            if (debug > 1): print( ':: eol =', arg.dim )

        # quit if no progress
        l = len(rem)
        if (l == length):
            if (debug > 1): print( ':: error: no progress' )
            arg.dim += '; ERROR PARSING (3)'
            break
        length = l
    # end
# end

# ------------------------------------------------------------------------------
# Parses an arg's description for "arg >= ..." giving lower bounds.
# Tries to find all bounds that exist.
# Returns string of all bounds, seperated by ";".
def parse_lbound( arg ):
    if (debug): print( 'parse_lbound' )

    txt = arg.desc
    lbound = ''
    found = True
    while (found):
        found = False
        s = re.search( arg.name + ' *>= *max([^\n]*)(.*)', txt, re.IGNORECASE | re.DOTALL )
        if (s):
            expr = s.group(1)
            (b, r) = extract_bracketed( expr, '(', ')' )
            if (b):
                lbound = join( lbound, 'max( ' + b + ' )' )
                found = True
            txt = r + s.group(2)
        else:
            s = re.search( arg.name + ' *>= *([^,;.]+)(.*)', txt, re.IGNORECASE | re.DOTALL )
            if (s):
                lbound = join( lbound, s.group(1) )
                found = True
                txt = s.group(2)
    # end
    lbound = re.sub( r',(\S)', r', \1', lbound )
    arg.lbound = lbound.lower()
# end

# ------------------------------------------------------------------------------
# state tracks what is looked for next:
#   function,
#   purpose,
#   param, "verbatim", param's description [repeated],
#   ingroup
i = 0
state_func          = i; i += 1

state_purpose       = i; i += 1
state_purpose_verb  = i; i += 1
state_purpose_desc  = i; i += 1

state_param         = i; i += 1
state_param_verb    = i; i += 1
state_param_desc    = i; i += 1

state_details       = i; i += 1
state_details_verb  = i; i += 1
state_details_desc  = i; i += 1

state_further_verb  = i; i += 1

state_done          = i; i += 1

# ------------------------------------------------------------------------------
# Parses a single LAPACK src file, reading its function arguments.
# Returns a string with the wrapper.
def parse_lapack( path ):
    (d, filename) = os.path.split( path )
    (funcname, ext) = os.path.splitext( filename )
    func   = Func( funcname )
    state  = state_func
    arg    = None

    # --------------------
    # parse Fortran doxygen for purpose and params
    f = open( path )
    for line in f:
        if (state == state_func):
            s = re.search( r'^\* +\b(.*)\b +(?:recursive +)?function +(\w+)', line, re.IGNORECASE )  #\( *(.*) *\)', line )
            if (s):
                func.retval = typemap[ s.group(1).lower() ]
                func.is_func = True
                f2 = s.group(2).lower()
                if (func.xname != f2):
                    print( 'Warning: filename', filename, "doesn't match function", f2 )
                state = state_purpose
                continue
            # end

            s = re.search( r'^\* +(?:recursive +)?subroutine +(\w+)', line, re.IGNORECASE )  #\( *(.*) *\)', line )
            if (s):
                f2 = s.group(1).lower()
                if (func.xname != f2):
                    print( 'Warning: filename', filename, "doesn't match function", f2 )
                state = state_purpose
            # end

        # ----------
        elif (state == state_purpose):
            if (re.search( r'^\*> *\\par +Purpose:', line )):
                state = state_purpose_verb

        elif (state == state_purpose_verb):
            if (re.search( r'^\*> *\\verbatim', line )):
                state = state_purpose_desc

        elif (state == state_purpose_desc):
            if (re.search( r'^\*> *\\endverbatim', line )):
                state = state_param
            else:
                func.purpose += line

        # ----------
        elif (state == state_param):
            s = re.search( r'^\*> *\\param\[(in|out|in,out|inout)\] +(\w+)', line )
            if (s):
                intent = s.group(1).lower()
                var    = s.group(2)
                arg = Arg( var, intent )
                func.args.append( arg )
                state = state_param_verb
                if (debug): print( 'param:', intent, var )
                continue
            # end

            # get group minus precision (sdcz prefix)
            s = re.search( r'^\*> *\\ingroup +(?:[sdcz](\w+)|(\w+))', line )
            if (s):
                func.group = s.group(1) or s.group(2)
                state = state_details
                #print( 'details head (1)' )
                continue

            # shouldn't happen, but if \\ingroup isn't before Further Details
            if (re.search( r'^\*> *\\par Further Details', line )):
                state = state_further_verb
                #print( 'details head (1)' )
                continue

            if (re.search( r'^      SUBROUTINE', line )):
                # finished!
                #print( 'finished (1)' )
                break

        elif (state == state_param_verb):
            if (re.search( r'^\*> *\\verbatim', line )):
                state = state_param_desc

        elif (state == state_param_desc):
            if (re.search( r'^\*> *\\endverbatim', line )):
                state = state_param
                arg = None
            else:
                arg.desc += line

        # ----------
        elif (state == state_details):
            #print( line )
            if (re.search( r'^\*> *\\par Further Details', line )):
                state = state_details_verb
                #print( 'details head (2)' )
            elif (re.search( r'^      SUBROUTINE', line )):
                # finished!
                #print( 'finished (2)' )
                break

        elif (state == state_details_verb):
            if (re.search( r'^\*> *\\verbatim', line )):
                state = state_details_desc
                #func.details += '### Further Details\n'
                #print( 'details verb' )

        elif (state == state_details_desc):
            if (re.search( r'^\*> *\\endverbatim', line )):
                # finished!
                #print( 'finished (3)' )
                break
            else:
                func.details += line
        # end
    # end

    # strip comment chars; squeeze space
    func.purpose = re.sub( r'\*> ?',   r'',    func.purpose )
    func.purpose = re.sub( r'(\S)  +', r'\1 ', func.purpose )

    # strip comment chars
    # LAPACK details tend to be indented 2 spaces instead of 1 space
    func.details = re.sub( r'\*> {0,2}', r'', func.details )
    # for now, don't squeeze space (destroys examples like in gebrd)
    #func.details = re.sub( r'(\S)  +', r'\1 ', func.details )

    if (func.name in arg_rename):
        for (search, replace) in arg_rename[ func.name ]:
            func.purpose = re.sub( r'\b' + search + r'\b', replace, func.purpose )

    # --------------------
    # parse arguments for properties
    i = 0
    for arg in func.args:
        if (debug): print( '-'*40 + '\narg:', arg.name )

        # strip comment chars; squeeze space
        # normally, lapack indents 10 spaces; if more, leave excess
        arg.desc = re.sub( r'\*> {0,10}', r'',    arg.desc )
        arg.desc = re.sub( r'(\S)  +',    r'\1 ', arg.desc )

        # extract data type and if array
        s = (re.search( r'^ *' + arg.name + ' +is +(CHARACTER|INTEGER|REAL|DOUBLE PRECISION|COMPLEX\*16|COMPLEX|LOGICAL)( array)?', arg.desc ) or
             re.search( r'^ *\(workspace\) +(CHARACTER|INTEGER|REAL|DOUBLE PRECISION|COMPLEX\*16|COMPLEX|LOGICAL)( array)?', arg.desc ) or
             re.search( r'^ *' + arg.name + ' +is a (LOGICAL FUNCTION of (?:one|two|three) (?:REAL|DOUBLE PRECISION|COMPLEX\*16|COMPLEX) arguments?)( array)?', arg.desc ))
        if (s):
            arg.dtype = typemap[ s.group(1).lower() ]
            arg.is_array = (s.group(2) == ' array')
        else:
            print( 'ERROR: unknown dtype:\n' + arg.desc )

        # normalize names
        # todo: in all desc?
        if (arg.name == 'CWORK'):
            arg.name = 'WORK'
            arg.desc = re.sub( r'\bCWORK\b', r'WORK', arg.desc )

        if (arg.name == 'SELCTG'):
            arg.name = 'SELECT'
            arg.desc = re.sub( r'\bSELCTG\b', r'SELECT', arg.desc )

        if (arg.name == 'BALANC'):
            arg.name = 'BALANCE'
            arg.desc = re.sub( r'\bBALANC\b', r'BALANCE', arg.desc )

        if (func.name in arg_rename):
            for (search, replace) in arg_rename[ func.name ]:
                arg.desc = re.sub( r'\b' + search + r'\b', replace, arg.desc )
                #print( 'func', func.name, 'arg', arg.name, 'search', search, 'replace', replace, '\n<<<\n', arg.desc, '\n>>>\n' )
                if (arg.name == search):
                    arg.name = replace
                    arg.name_orig = replace

        # lowercase scalars, work and pivot arrays
        if (not arg.is_array or arg.name in
                ('WORK', 'SWORK', 'RWORK', 'IWORK', 'BWORK',
                'IPIV', 'JPIV', 'JPVT', 'PIV',
                'BERR', 'FERR', 'SELECT',
                'TAU', 'TAUP', 'TAUQ', 'TAUA', 'TAUB',
                'ISUPPZ', 'IFAIL', 'ISEED', 'IBLOCK', 'ISPLIT',
                'SCALE', 'LSCALE', 'RSCALE',
                'ALPHA', 'ALPHAR', 'ALPHAI',
                'BETA', 'BETAR', 'BETAI',
                'THETA', 'PHI',
                'HOUS2',
                'ERR_BNDS_NORM', 'ERR_BNDS_COMP', 'PARAMS',
                'RCONDE', 'RCONDV')):
            arg.name = arg.name.lower()

        # mark work, lwork, ldwork, etc.
        if (re.search( r'^[crsib]?work$', arg.name )):
            arg.is_work = True
        elif (re.search( r'^(l|ld)[crsib]?work$', arg.name )):
            arg.is_lwork = True

        # iwork, bwork needs lapack_int, not int64
        if (arg.name in ('iwork', 'bwork')):
            arg.dtype = 'lapack_int'

        # extract array dimensions or scalar lower bounds
        if (arg.is_array):
            parse_dim( arg )
        else:
            parse_lbound( arg )

        # check for workspace query
        if (arg.is_lwork):
            s = re.search( r'If ' + arg.name + ' *= *-1.*workspace query', arg.desc, re.DOTALL | re.IGNORECASE )
            if (s):
                func.args[i-1].use_query = True  # work
                arg.use_query = True  # lwork
        # end

        # lname = local name
        # pname = pointer name to pass to Fortran
        arg.lname = arg.name
        arg.pname = arg.name
        if (arg.intent == 'in' and not arg.is_array and not re.search( r'LAPACK_._SELECT', arg.dtype)):
            # scalars, dimensions, enums
            if (arg.dtype in ('int64_t', 'bool', 'char')):
                # dimensions, enums
                arg.lname += '_'
            arg.pname = '&' + arg.lname
        elif (arg.intent in ('out', 'in,out') and not arg.is_array and arg.dtype in ('int64_t', 'bool')):
            # output dimensions (nfound, rank, ...)
            arg.lname += '_'
            arg.pname = '&' + arg.lname
        elif (arg.is_array and arg.dtype in ('int64_t', 'bool')):
            # integer or bool arrays: ipiv, select, etc.
            arg.lname += '_'
            arg.pname = arg.name + '_ptr'
        elif (arg.is_work):
            arg.pname = '&' + arg.name + '[0]'
        # end

        # map char to enum (after doing lname)
        if (arg.dtype == 'char'):
            arg.is_enum = True
            try:
                arg.dtype = enum_map[ arg.name ][0]
            except Exception as ex:
                arg.dtype = 'UNKNOWN'
                print( 'ERROR: unknown enum', arg.name )

        if (debug):
            print(   'arg       = ' + arg.name +
                   '\ndtype     = ' + arg.dtype +
                   '\nintent    = ' + arg.intent +
                   '\nis_array  = ' + str(arg.is_array) +
                   '\nis_work   = ' + str(arg.is_work) +
                   '\nis_lwork  = ' + str(arg.is_lwork) +
                   '\nuse_query = ' + str(arg.use_query))
            if (arg.is_array):
                print( 'dim       = ' + arg.dim.lower() )
            else:
                print( 'lbound    = ' + arg.lbound.lower() )
        # end
        if (debug > 1):
            print( '\ndescription =\n' + arg.desc )
        # end

        i += 1
    # end
    return func
# end

# ------------------------------------------------------------------------------
# regexp match:
# \b([A-Z0-9]{2,})  (\s*(?:=|!=)\s*)  '(\w)'  (?:(\s+or\s+)  '(\w)')?
#   enum            oper = or !=       val       conjunction  val2     <= groups
#
# converts: UPLO = 'U' or 'u'
# to        uplo = Upper
def sub_enum_short( match ):
    try:
        (enum, oper, val, conj, val2) = match.groups()
        enum = enum.lower()
        val  = val.lower()
        char2enum = enum_map[ enum ][2]
        txt = enum + oper + char2enum[ val ]
        if (val2):
            val2 = val2.lower()
            if (char2enum[ val ] != char2enum[ val2 ]):
                txt += conj + char2enum[ val2 ]
        return txt
    except Exception as ex:
        print( 'ERROR: unknown enum value, groups: ' + str(match.groups())
               + '; exception: ' + str(ex) )
        return match.group(0) + ' [TODO: unknown enum]'
# end

# ------------------------------------------------------------------------------
# regexp match:
# ^ *= *'(\w)' (?:(\s+or\s+)  '(\w)')?[:,]
#        val      conjunction  val2         <= groups
#
# converts:  = 'U' or 'u':
# to:        - lapack::Uplo::Upper:
def sub_enum_list( arg, match ):
    try:
        (val, conj, val2) = match.groups()
        val = val.lower()
        enum_class = enum_map[ arg.name ][0]
        char2enum  = enum_map[ arg.name ][2]
        txt = '- ' + enum_class + '::' + char2enum[ val ]
        if (val2):
            val2 = val2.lower()
            if (char2enum[ val ] != char2enum[ val2 ]):
                txt += conj + char2enum[ val2 ]
        return txt + ':'
    except Exception as ex:
        print( 'ERROR: unknown enum value, arg: ' + arg.name + ', groups: '
               + str(match.groups()) + '; exception: ' + str(ex) )
        return match.group(0) + ' [TODO: unknown enum]'
# end

# ------------------------------------------------------------------------------
def parse_docs( func, txt, variables, indent=0 ):
    # exponents
    txt = re.sub( r'\*\*', r'^', txt )

    # Fortran operators:  ".LT."  =>  "<"
    txt = re.sub( r'\.(LT|LE|GT|GE|EQ|NE)\.',
                  lambda match: fortran_operators[ match.group(1) ], txt, 0, re.I )

    # space around operators
    txt = re.sub( r'(\S)(<=|>=)(\S)', r'\1 \2 \3', txt )
    txt = re.sub( r'(\S)(<=|>=)(\S)', r'\1 \2 \3', txt )

    # make indents 4 spaces (lapack usually has 3)
    txt = re.sub( r'^   +', r'    ', txt, 0, re.M )

    # convert enums, e.g.,
    # "UPLO = 'U'" => "uplo = Upper"
    txt = re.sub( r"\b([A-Z0-9_]{2,})(\s*(?:=|!=)\s*)'(\w)'(?:(\s+or\s+)'(\w)')?",
                  sub_enum_short, txt )

    # convert function names
    txt = re.sub( func.xname.upper(), '`'+ func.name + '`', txt )
    txt = re.sub( lapack_functions_re,
                  lambda match: '`lapack::' + match.group(1).lower() + '`', txt )

    # INFO references
    txt = re.sub( r'On exit, if INFO *= *0,', r'On successful exit,', txt, 0, re.I )
    txt = re.sub( r'(if) INFO *= *0', r'\1 successful', txt, 0, re.I )
    txt = re.sub( r'\bINFO *(=|>|>=) *', r'return value \1 ', txt )

    # rename arguments (lowercase, spelling)
    for (search, replace) in variables:
        txt = re.sub( search, replace, txt )

    # hyphenate m-by-n
    txt = re.sub( r'\b([a-z]) by ([a-z])\b', r'\1-by-\2', txt )

    # prefix lines
    txt = re.sub( r'^', r'/// ' + ' '*indent, txt, 0, re.M ) + '\n'
    return txt
# end

# ------------------------------------------------------------------------------
# returns C++ docs for given function
def generate_docs( func ):
    variables = []
    for arg in func.args:
        if (arg.name_orig != arg.name):
            variables.append( (r'\b' + arg.name_orig + r'\b', arg.name) )
    #print( 'variables', variables )

    txt = ''

    # --------------------
    d = func.purpose

    # strip function name and capitalize first word, e.g.,
    # "ZPOTRF computes ..." => "Computes ..."
    d = re.sub( r'^\s*[A-Z0-9_]+ (\w)', lambda s: s.group(1).upper(), d )
    txt += parse_docs( func, d, variables, indent=0 )
    txt += '''\
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
'''

    version = first_version[ func.name ]
    v = version[0]*10000 + version[1]*100 + version[2]
    if (v > 30201):
        txt += '/// @since LAPACK %d.%d.%d\n///\n' % (version)

    # --------------------
    i = 0
    for arg in func.args:
        if (arg.is_work or arg.is_lwork):
            continue

        d = arg.desc

        # remove "VAR is INTEGER" line
        d = re.sub( r'^ *' + arg.name.upper()
                    + ' +is +(CHARACTER|INTEGER|REAL|DOUBLE PRECISION|COMPLEX\*16|COMPLEX|LOGICAL).*\n',
                    r'', d )

        # change "Specifies the ..." => "The ..."
        d = re.sub( r'^ *' + 'Specifies (\w)',
                    lambda match: match.group(1).upper(), d )

        if (arg.name == 'info'):
            # @retval
            d = re.sub( r'^ *< *0: *if INFO *= *-i, the i-th argument.*\n', r'',
                        d, 0, re.M | re.I )
            d = re.sub( r'^ *(=|<|<=|>|>=) ', r'@retval \1 ', d, 0, re.M )

            # increase indented lines
            d = re.sub( r'^      ', r'             ', d, 0, re.M )

            txt += parse_docs( func, d, variables, indent=0 )
        else:
            txt += '/// @param[' + arg.intent + '] ' + arg.name + '\n'

            if (arg.is_array):
                s = re.search( r'^ *\( *([^(),]+), *([^(),]+) *\) *$', arg.dim )
                if (s):
                    rows = 'ROWS'
                    ld   = s.group(1)
                    cols = s.group(2)
                    dim = ld + '-by-' + cols
                    if (ld.startswith('ld') and i+1 < len(func.args)):
                        ld_arg = func.args[i+1]
                        if (ld_arg.lbound):
                            s = re.search( r'^ *max\( *1, *(\w+) *\)', ld_arg.lbound )
                            if (s):
                                rows = s.group(1)
                            else:
                                rows = ld_arg.lbound
                    # end
                    dim2 = rows + '-by-' + cols
                    d = 'The ' + dim2 + ' matrix ' + arg.name + ', stored in an ' \
                      + dim + ' array.\n' + d
                    # [dimension ' + arg.dim + ']
                else:
                    # try to strip off one set of parens
                    (dim, rem) = extract_bracketed( arg.dim, '(', ')' )
                    if (txt and rem == ''):
                        d = 'The vector ' + arg.name + ' of length ' + dim + '.\n' + d
                    else:
                        d = 'The vector ' + arg.name + ' of length ' + arg.dim + '.\n' + d
            # end

            # convert enum values, e.g.,
            # = 'U': ...    =>    - Upper: ...
            # = 'L': ...    =>    - Lower: ...
            if (arg.is_enum):
                d = re.sub( r"^ *= *'(\w)'(?:(\s+or\s+)'(\w)')?[:,]",
                            lambda match: sub_enum_list( arg, match ),
                            d, 0, re.M )
            # end

            d = parse_docs( func, d, variables, indent=4 )

            # add "\n" on blank lines
            d = re.sub( r'^(/// +)\n///', r'\1\\n\n///', d, 0, re.M )
            txt += d
        # end
        i += 1
    # end

    # Further Details section
    d = func.details
    if (d):
        txt += '//------------------------------------------------------------------------------\n'
        txt += '/// @par Further Details\n'
        txt += parse_docs( func, d, variables, indent=4 )
    # end

    txt += '/// @ingroup ' + func.group + '\n'

    # post renaming
    if (func.name in post_rename):
        for (search, replace) in post_rename[ func.name ]:
            txt = re.sub( search, replace, txt )

    # trim trailing whitespace
    txt = re.sub( r' +$', r'', txt, 0, re.M )

    return txt
# end

# ------------------------------------------------------------------------------
# returns LAPACK++ wrapper for given function
def generate_wrapper( func, header=False ):
    # --------------------
    # build list of arguments for prototype, query, and call
    int_checks = ''
    local_vars = ''
    alloc_work = ''
    query      = ''
    cleanup    = ''
    info_throw = ''
    info_return = ''
    proto_args = []
    query_args = []
    call_args  = []
    alias_args = []
    use_query  = False
    i = 0
    for arg in func.args:
        # arg.name  is base name, like "m" or "A"
        # arg.lname is local name, like "m_"
        # arg.pname is pointer name to pass to Fortran, like "&m_" or "A"
        ##print( 'arg ' + arg.name + ', pname ' + arg.pname + ', dtype ' + arg.dtype )

        # arrays start new line
        prefix = ''
        if (arg.is_array):
            prefix = '\n' + tab*2

        # cast complex pointers
        cast = ''
        m = re.search( '^std::complex<(\w+)>$', arg.dtype )
        if (m):
            cast = '(lapack_complex_' + m.group(1) + '*) '
        call_args.append( prefix + cast + arg.pname )

        if (arg.intent == 'in'):
            if (arg.is_array):
                # input arrays
                proto_args.append( '\n    ' + arg.dtype + ' const* ' + arg.name )
                alias_args.append( arg.name )
                query_args.append( prefix + cast + arg.pname )
                if (arg.dtype in ('int64_t', 'bool')):
                    # integer input arrays: copy in input
                    local_vars += (tab + '#if 1\n'
                               +   tab*2 + '// 32-bit copy\n'
                               +   tab*2 + 'std::vector< lapack_int > ' + arg.lname + '( &' + arg.name + '[0], &' + arg.name + '[' + arg.dim + '] );\n'
                               +   tab*2 + 'lapack_int const* ' + arg.pname + ' = &' + arg.lname + '[0];\n'
                               +   tab + '#else\n'
                               +   tab*2 + 'lapack_int const* ' + arg.pname + ' = ' + arg.lname + ';\n'
                               +   tab + '#endif\n')
                # end
            elif (arg.use_query):
                # lwork, lrwork, etc. local variables; not in proto_args
                query_args.append( '&ineg_one' )
                use_query = True
            elif (arg.is_lwork):
                # lwork, etc. local variables; not in proto_args
                local_vars += tab + 'lapack_int ' + arg.lname + ' = (lapack_int) ' + arg.lbound + ';\n'
            else:
                proto_args.append( arg.dtype + ' ' + arg.name )
                alias_args.append( arg.name )
                query_args.append( prefix + cast + arg.pname )
                if (arg.dtype in ('int64_t', 'bool')):
                    # local 32-bit copy of 64-bit int
                    int_checks += tab*2 + 'lapack_error_if( std::abs(' + arg.name + ') > std::numeric_limits<lapack_int>::max() );\n'
                    local_vars += tab + 'lapack_int ' + arg.lname + ' = (lapack_int) ' + arg.name + ';\n'
                elif (arg.is_enum):
                    enum2char = enum_map[ arg.name ][1]
                    local_vars += tab + 'char ' + arg.lname + ' = ' + enum2char + '( ' + arg.name + ' );\n'
                # end
            # end
        else:
            if (arg.is_array and arg.is_work):
                # work, rwork, etc. local variables; not in proto_args
                query_args.append( prefix + cast + 'qry_' + arg.name )
                query += tab + arg.dtype + ' qry_' + arg.name + '[1];\n'

                if (arg.use_query):
                    alloc_work += tab + 'std::vector< ' + arg.dtype + ' > ' + arg.lname + '( ' + func.args[i+1].lname + ' );\n'
                else:
                    alloc_work += tab + 'std::vector< ' + arg.dtype + ' > ' + arg.lname + '( ' + arg.dim.lower() + ' );\n'

                ##alloc_work += (tab + arg.dtype + '* ' + arg.name + '_'
                ##           +   ' = new ' + arg.dtype + '[ l' + arg.name + ' ];\n')
                ##free_work  +=  tab + 'delete[] ' + arg.name + '_;\n'
            elif (not arg.is_array and arg.dtype in ('int64_t', 'bool')):
                # output dimensions (nfound, etc.)
                query_args.append( prefix + cast + arg.pname )
                if (arg.name == 'info'):
                    # info not in proto_args
                    local_vars += tab + 'lapack_int ' + arg.lname + ' = 0;\n'
                    func.retval = 'int64_t'
                    info_throw = (tab + 'if (info_ < 0) {\n'
                               +  tab*2 + 'throw Error();\n'
                               +  tab + '}\n')
                    info_return = tab + 'return info_;\n'
                else:
                    proto_args.append( '\n    ' + arg.dtype + '* ' + arg.name )
                    alias_args.append( arg.name )
                    local_vars += tab + 'lapack_int ' + arg.lname + ' = (lapack_int) *' + arg.name + ';\n'
                    cleanup += tab + '*' + arg.name + ' = ' + arg.lname + ';\n'
            else:
                # output array
                query_args.append( prefix + cast + arg.pname )
                proto_args.append( '\n    ' + arg.dtype + '* ' + arg.name )
                alias_args.append( arg.name )
                if (arg.is_array and (arg.dtype in ('int64_t', 'bool'))):
                    if (arg.intent == 'in,out'):
                        # copy in input, copy out in cleanup
                        local_vars += (tab + '#if 1\n'
                                   +   tab*2 + '// 32-bit copy\n'
                                   +   tab*2 + 'std::vector< lapack_int > ' + arg.lname + '( &' + arg.name + '[0], &' + arg.name + '[' + arg.dim + '] );\n'
                                   +   tab*2 + 'lapack_int* ' + arg.pname + ' = &' + arg.lname + '[0];\n'
                                   +   tab + '#else\n'
                                   +   tab*2 + 'lapack_int* ' + arg.pname + ' = ' + arg.name + ';\n'
                                   +   tab + '#endif\n')
                    else:
                        # allocate w/o copy, copy out in cleanup
                        local_vars += (tab + '#if 1\n'
                                   +   tab*2 + '// 32-bit copy\n'
                                   +   tab*2 + 'std::vector< lapack_int > ' + arg.lname + '( ' + arg.dim + ' );\n'
                                   +   tab*2 + 'lapack_int* ' + arg.pname + ' = &' + arg.lname + '[0];\n'
                                   +   tab + '#else\n'
                                   +   tab*2 + 'lapack_int* ' + arg.pname + ' = ' + arg.name + ';\n'
                                   +   tab + '#endif\n')
                    # end
                    cleanup += (tab + '#if 1\n'
                            +   tab*2 + 'std::copy( ' + arg.lname + '.begin(), ' + arg.lname + '.end(), ' + arg.name + ' );\n'
                            +   tab + '#endif\n')
            # end
        # end
        i += 1
    # end

    # --------------------
    # build query
    if (use_query):
        query =  ('\n'
              +   tab + '// query for workspace size\n'
              +   query
              +   tab + 'lapack_int ineg_one = -1;\n'
              +   tab + 'LAPACK_' + func.xname + '(\n' + tab*2 + ', '.join( query_args ) + ' );\n'
              +   tab + 'if (info_ < 0) {\n'
              +   tab*2 + 'throw Error();\n'
              +   tab + '}\n')
        # assume when arg is l*work, last will be *work
        last = None
        for arg in func.args:
            if (arg.use_query and not arg.is_array):
                query += tab + 'lapack_int ' + arg.name + '_ = real(qry_' + last.name + '[0]);\n'
            last = arg
        # end
    else:
        query = ''
    # end

    if (int_checks):
        int_checks = (tab + '// check for overflow\n'
                   +  tab + 'if (sizeof(int64_t) > sizeof(lapack_int)) {\n'
                   +  int_checks
                   +  tab + '}\n')
    # end

    if (alloc_work):
        alloc_work = ('\n'
                   +  tab + '// allocate workspace\n'
                   +  alloc_work)
    # end

    if (func.is_func):
        call = '\n' + tab + 'return LAPACK_' + func.xname + '(\n' + tab*2 + ', '.join( call_args ) + ' );\n'
        if (info_throw or cleanup or info_return):
            print( 'Warning: why is info or cleanup on a function?' )
    else:
        call = '\n' + tab + 'LAPACK_' + func.xname + '(\n' + tab*2 + ', '.join( call_args ) + ' );\n'

    if (header):
        txt = (func.retval + ' ' + func.name + '(\n'
            +  tab + ', '.join( proto_args )
            +  ' );\n\n')
        # aliases for real routines (hesv => sysv, etc.)
        s = re.search( r'^[sd](sy|sp|sb|or|op)(\w+)', func.xname )
        if (s):
            alias = alias_map[ s.group(1) ] + s.group(2)
            # todo: he
            txt += ('// ' + alias + ' alias to ' + func.name + '\n'
                +   'inline ' + func.retval + ' ' + alias + '(\n'
                +   tab + ', '.join( proto_args ) + ' )\n'
                +   '{\n'
                +   tab + 'return ' + func.name + '( ' + ', '.join( alias_args ) + ' );\n'
                +   '}\n\n')
        # end
    else:
        txt = (func.retval + ' ' + func.name + '(\n'
            +  tab + ', '.join( proto_args )
            +  ' )\n{\n'
            +  int_checks
            +  local_vars
            +  query
            +  alloc_work
            +  call
            +  info_throw
            +  cleanup
            +  info_return
            +  '}\n\n')
    # end

    # post renaming
    if (func.name in post_rename):
        for (search, replace) in post_rename[ func.name ]:
            txt = re.sub( search, replace, txt )

    # trim trailing whitespace
    txt = re.sub( r' +$', r'', txt, 0, re.M )

    return txt
# end

# ------------------------------------------------------------------------------
def generate_tester( funcs ):
    # heuristic:
    # assume ints before 1st array are input parameters (trans, m, n, ...)
    # while  ints after  1st array are derived (lda, ...)
    pre_arrays = True

    scalars  = ''  # pre-arrays
    scalars2 = ''  # post-arrays
    sizes    = ''
    arrays   = ''
    init     = (tab + 'int64_t idist = 1;\n'
             +  tab + 'int64_t iseed[4] = { 0, 1, 2, 3 };\n')
    copy     = ''
    flop_args = [] # arguments for calling Gflops::foo(), pre-array
    tst_args = []  # arguments for calling C++ test
    ref_args = []  # arguments for calling Fortran reference
    verify   = ''

    func = funcs[-1]  # guess last one is complex, to get real_t below
    for arg in func.args:
        if (not (arg.is_work or arg.is_lwork or arg.name == 'info')):
            # determine template type from data type
            if (arg.dtype in ('float', 'double')):
                if (re.search( r'^[cz]', func.xname )):
                    arg.ttype = 'real_t'
                else:
                    arg.ttype = 'scalar_t'
                arg.ttype_ref = arg.ttype
            elif (arg.dtype in ('std::complex<float>', 'std::complex<double>')):
                arg.ttype = 'scalar_t'
                arg.ttype_ref = arg.ttype
            elif (arg.dtype in ('int64_t')):
                arg.ttype = arg.dtype
                arg.ttype_ref = 'lapack_int'
            else:
                arg.ttype = arg.dtype
                arg.ttype_ref = arg.ttype

            if (arg.is_array):
                pre_arrays = False

                # look for 2-D arrays: (m,n), change to: m * n
                s = re.search( r'^ *\( *(\w+), *(\w+) *\) *$', arg.dim )
                if (s):
                    dim = s.group(1) + ' * ' + s.group(2)
                else:
                    dim = arg.dim

                sizes += tab + 'size_t size_' + arg.name + ' = (size_t) ' + dim + ';\n';
                if (arg.intent == 'in' and not arg.dtype in ('int64_t')):
                    arrays += tab + 'std::vector< ' + arg.ttype + ' > ' + arg.name + '( size_' + arg.name + ' );\n'
                    if (arg.dtype in ('float', 'double', 'std::complex<float>', 'std::complex<double>')):
                        init += tab + 'lapack::larnv( idist, iseed, ' + arg.name + '.size(), &' + arg.name + '[0] );\n'
                    else:
                        init += tab + '// todo: initialize ' + arg.name
                    tst_args.append( '&' + arg.name + '[0]' )
                    ref_args.append( '&' + arg.name + '[0]' )
                else:
                    arrays += (tab + 'std::vector< ' + arg.ttype     + ' > ' + arg.name + '_tst( size_' + arg.name + ' );\n'
                           +   tab + 'std::vector< ' + arg.ttype_ref + ' > ' + arg.name + '_ref( size_' + arg.name + ' );\n')
                    if ('in' in arg.intent):
                        if (arg.dtype in ('float', 'double', 'std::complex<float>', 'std::complex<double>')):
                            init += tab + 'lapack::larnv( idist, iseed, ' + arg.name + '_tst.size(), &' + arg.name + '_tst[0] );\n'
                            copy += tab + arg.name + '_ref = ' + arg.name + '_tst;\n'
                        elif ('in' in arg.intent):
                            init += tab + '// todo: initialize ' + arg.name + '_tst and ' + arg.name + '_ref\n'
                    # end
                    tst_args.append( '&' + arg.name + '_tst[0]' )
                    ref_args.append( '&' + arg.name + '_ref[0]' )
                    if ('out' in arg.intent):
                        verify += tab*2 + 'error += abs_error( ' + arg.name + '_tst, ' + arg.name + '_ref );\n'
                # end
            else:
                # not array
                if (arg.intent == 'in'):
                    if (pre_arrays):
                        if (arg.name not in ('uplo')):
                            flop_args.append( arg.name )
                        if (arg.name in ('m', 'n', 'k')):
                            scalars += tab + arg.ttype + ' ' + arg.name + ' = params.dim.' + arg.name + '();\n'
                        else:
                            scalars += tab + arg.ttype + ' ' + arg.name + ' = params.' + arg.name + '();\n'
                    elif (arg.lbound):
                        if (arg.name.startswith('ld')):
                            scalars2 += tab + arg.ttype + ' ' + arg.name + ' = roundup( ' + arg.lbound + ', align );\n'
                        else:
                            scalars2 += tab + arg.ttype + ' ' + arg.name + ' = ' + arg.lbound + ';\n'
                    elif ('in' in arg.intent):
                        scalars2 += tab + arg.ttype + ' ' + arg.name + ';  // todo value\n'
                    else:
                        scalars2 += tab + arg.ttype + ' ' + arg.name + ';\n'
                    tst_args.append( arg.name )
                    if (arg.is_enum):
                        # convert enum to char, e.g., uplo2char(uplo)
                        enum2char = enum_map[ arg.name ][1]
                        #s.group(1).lower() +
                        ref_args.append( enum2char + '(' + arg.name + ')' )
                    else:
                        ref_args.append( arg.name )
                else:
                    if (pre_arrays):
                        scalars += tab + arg.ttype + ' ' + arg.name + '_tst = params.' + arg.name + '();\n'
                        scalars += tab + arg.ttype + ' ' + arg.name + '_ref = params.' + arg.name + '();\n'
                    elif (arg.lbound):
                        scalars2 += tab + arg.ttype + ' ' + arg.name + '_tst = ' + arg.lbound + ';\n'
                        scalars2 += tab + arg.ttype + ' ' + arg.name + '_ref = ' + arg.lbound + ';\n'
                    elif ('in' in arg.intent):
                        scalars2 += tab + arg.ttype + ' ' + arg.name + '_tst;  // todo value\n'
                        scalars2 += tab + arg.ttype + ' ' + arg.name + '_ref;  // todo value\n'
                    else:
                        scalars2 += tab + arg.ttype + ' ' + arg.name + '_tst;\n'
                        scalars2 += tab + arg.ttype + ' ' + arg.name + '_ref;\n'
                    tst_args.append( '&' + arg.name + '_tst' )
                    ref_args.append( '&' + arg.name + '_ref' )
                    verify += tab*2 + 'error += std::abs( ' + arg.name + '_tst - ' + arg.name + '_ref );\n'
                # end
            # end
        # end
    # end

    lapacke  = ('//------------------------------------------------------------------------------\n'
             +  '// Simple overloaded wrappers around LAPACKE (assuming routines in LAPACKE).\n'
             +  '// These should go in test/lapacke_wrappers.hh.\n' )
    for func in funcs:
        lapacke_proto = []
        lapacke_args  = []
        for arg in func.args:
            if (arg.name in ('info', 'work', 'rwork', 'iwork', 'lwork', 'lrwork', 'ldwork', 'liwork')):
                continue
            elif (arg.is_enum):
                lapacke_proto.append( 'char ' + arg.name )
            elif (arg.dtype in ('int64_t')):
                if (arg.is_array):
                    lapacke_proto.append( '\n' + tab + 'lapack_int* ' + arg.name )
                else:
                    lapacke_proto.append( 'lapack_int ' + arg.name )
            else:
                if (arg.is_array or ('out' in arg.intent)):
                    lapacke_proto.append( '\n' + tab + arg.dtype + '* ' + arg.name )
                else:
                    lapacke_proto.append( arg.dtype + ' ' + arg.name )
            # end

            pre = ''
            if (arg.is_array or ('out' in arg.intent)):
                pre = '\n' + tab*2
            # cast complex pointers
            m = re.search( '^std::complex<(\w+)>$', arg.dtype )
            if (m):
                pre += '(lapack_complex_' + m.group(1) + '*) '
            lapacke_args.append( pre + arg.name )
        # end
        lapacke_proto = ', '.join( lapacke_proto )
        lapacke_args  = ', '.join( lapacke_args )
        lapacke += ('inline lapack_int LAPACKE_' + func.name + '(\n'
                +   tab + lapacke_proto + ' )\n'
                +   '{\n'
                +   tab + 'return LAPACKE_' + func.xname + '(\n' + tab*2 + 'LAPACK_COL_MAJOR, ' + lapacke_args + ' );\n'
                +   '}\n\n')
    # end

    flop_args = ', '.join( flop_args )
    tst_args = ', '.join( tst_args )
    ref_args = ', '.join( ref_args )

    #----------
    version = first_version[ func.name ]
    v = version[0]*10000 + version[1]*100 + version[2]
    if (v > 30201):
        # requires_if2 ... requires_end  goes around test_xyz_work routine.
        # requires_if  ... requires_else goes inside test_xyz dispatch routine.
        requires_if   = '#if LAPACK_VERSION >= %d%02d%02d  // >= %d.%d.%d\n' % (version + version)
        requires_if2  = requires_if + '\n'
        requires_end  = '\n#endif  // LAPACK >= %d.%d.%d\n' % (version)
        requires_else = '''\
#else
    fprintf( stderr, "''' + func.name + ''' requires LAPACK >= %d.%d.%d\\n\\n" );
    exit(0);
#endif
''' % (version)
    else:
        requires_if   = ''
        requires_if2  = ''
        requires_end  = ''
        requires_else = ''
    # end

    #----------
    # todo: only dispatch for precisions in funcs
    dispatch = ('''
//------------------------------------------------------------------------------
void test_''' + func.name + '''( Params& params, bool run )
{
''' + requires_if + '''\
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_''' + func.name + '''_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_''' + func.name + '''_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_''' + func.name + '''_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_''' + func.name + '''_work< std::complex<double> >( params, run );
            break;
    }
''' + requires_else + '''\
}
''')

    txt = (requires_if2
        +  lapacke
        +  '//------------------------------------------------------------------------------\n'
        +  'template< typename scalar_t >\n'
        +  'void test_' + func.name + '_work( Params& params, bool run )\n'
        +  '{\n'
        +  tab + 'using real_t = blas::real_type< scalar_t >;\n'
        +  '\n'
        +  tab + '// get & mark input values\n'
        +  scalars
        +  tab + 'int64_t align = params.align();\n'
        #+ tab + 'int64_t verbose = params.verbose();\n'
        +  '\n'
        +  tab + '// mark non-standard output values\n'
        +  tab + 'params.ref_time();\n'
        +  tab + 'params.ref_gflops();\n'
        +  tab + 'params.gflops();\n'
        +  '\n'
        +  tab + 'if (! run)\n'
        +  tab*2 + 'return;\n'
        +  '\n'
        +  tab + '//---------- setup\n'
        +  scalars2
        +  sizes
        +  '\n'
        +  arrays
        +  '\n'
        +  init
        +  copy
        +  '\n'
        +  tab + '//---------- run test\n'
        +  tab + 'testsweeper::flush_cache( params.cache() );\n'
        +  tab + 'double time = testsweeper::get_wtime();\n'
        +  tab + 'int64_t info_tst = lapack::' + func.name + '( ' + tst_args + ' );\n'
        +  tab + 'time = testsweeper::get_wtime() - time;\n'
        +  tab + 'if (info_tst != 0) {\n'
        +  tab + '    fprintf( stderr, "lapack::' + func.name + ' returned error %lld\\n", llong( info_tst ) );\n'
        +  tab + '}\n'
        +  '\n'
        +  tab + 'params.time() = time;\n'
        +  tab + 'double gflop = lapack::Gflop< scalar_t >::' + func.name + '( ' + flop_args + ' );\n'
        +  tab + 'params.gflops() = gflop / time;\n'
        +  '\n'
        +  tab + "if (params.ref() == 'y' || params.check() == 'y') {\n"
        +  tab*2 + '//---------- run reference\n'
        +  tab*2 + 'testsweeper::flush_cache( params.cache() );\n'
        +  tab*2 + 'time = testsweeper::get_wtime();\n'
        +  tab*2 + 'int64_t info_ref = LAPACKE_'  + func.name + '( ' + ref_args + ' );\n'
        +  tab*2 + 'time = testsweeper::get_wtime() - time;\n'
        +  tab*2 + 'if (info_ref != 0) {\n'
        +  tab*2 + '    fprintf( stderr, "LAPACKE_' + func.name + ' returned error %lld\\n", llong( info_ref ) );\n'
        +  tab*2 + '}\n'
        +  '\n'
        +  tab*2 + 'params.ref_time() = time;\n'
        +  tab*2 + 'params.ref_gflops() = gflop / time;\n'
        +  '\n'
        +  tab*2 + '//---------- check error compared to reference\n'
        +  tab*2 + 'real_t error = 0;\n'
        +  tab*2 + 'if (info_tst != info_ref) {\n'
        +  tab*2 + '    error = 1;\n'
        +  tab*2 + '}\n'
        +  verify
        +  tab*2 + 'params.error() = error;\n'
        +  tab*2 + 'params.okay() = (error == 0);  // expect lapackpp == lapacke\n'
        +  tab + '}\n'
        +  '}\n'
        +  requires_end
        +  dispatch
    )

    # post renaming
    if (func.name in post_rename):
        for (search, replace) in post_rename[ func.name ]:
            txt = re.sub( search, replace, txt )

    # trim trailing whitespace
    txt = re.sub( r' +$', r'', txt, 0, re.M )

    return txt
# end

# ------------------------------------------------------------------------------
# processes each routine (given as command line argument)
# to generate header, wrapper, or tester.
def process_routine( arg ):
    files = []
    for subdir in ('SRC', 'TESTING/MATGEN'):
        for p in ('s', 'd', 'c', 'z'):  #, 'ds', 'zc'):
            f = lapack + '/' + subdir + '/' + p + arg + ".f"
            if (os.path.exists( f )):
                files.append( f )
        # end
    # end

    if (not files):
        print( "no files found matching:", arg )
        return
    # end

    if (args.header):
        print( '// ' + '-'*77, file=header )

    if (args.wrapper):
        wrapper_file = os.path.join( gen, arg + '.cc' )
        print( 'generating', wrapper_file )
        wrapper = open( wrapper_file, 'w' )

    if (args.tester):
        tester_file = os.path.join( gen, 'test_' + arg + '.cc' )
        print( 'generating', tester_file )
        tester = open( tester_file, 'w' )
        print( tester_top, file=tester, end='' )

    requires_if  = ''
    requires_end = ''

    funcs = []
    i = 0
    for f in files:
        i += 1
        last = (i == len(files))
        print( '    ' + f )
        func = parse_lapack( f )
        funcs.append( func )

        if (i == 1):
            version = first_version[ func.name ]
            v = version[0]*10000 + version[1]*100 + version[2]
            if (v > 30201):
                requires_if  = '#if LAPACK_VERSION >= %d%02d%02d  // >= %d.%d.%d\n\n' % (version + version)
                requires_end = '\n#endif  // LAPACK >= %d.%d.%d\n' % (version)
            # end
            if (args.wrapper):
                print( wrapper_top1, file=wrapper, end='' )
                print( requires_if,  file=wrapper, end='' )
                print( wrapper_top2, file=wrapper, end='' )
            # end
        # end

        if (args.header):
            txt = generate_wrapper( func, header=True )
            print( txt, file=header, end='' )
        if (args.wrapper):
            txt = '// ' + '-'*77 + '\n'
            print( txt, file=wrapper, end='' )
            if (last):
                txt = generate_docs( func )
                print( txt, file=wrapper, end='' )
            else:
                print( '/// @ingroup ' + func.group + '\n', file=wrapper, end='' )
            txt = generate_wrapper( func )
            print( txt, file=wrapper, end='' )
    # end

    # tester needs to see all related functions
    if (args.tester):
        txt = generate_tester( funcs )
        print( txt, file=tester, end='' )

    if (args.wrapper):
        print( wrapper_bottom, file=wrapper, end='' )
        print( requires_end,   file=wrapper, end='' )
        wrapper.close()

    if (args.tester):
        print( tester_bottom, file=tester, end='' )
        tester.close()

    print()
# end

# ------------------------------------------------------------------------------
# Process each function on the command line, given as a base name like "getrf".
# Finds LAPACK functions with that base name and some precision prefix.
# Outputs to gen/basename.cc
lapack = os.environ['LAPACKDIR']

if (args.header):
    header_file = os.path.join( gen, 'wrappers.hh' )
    print( 'generating', header_file )
    header = open( header_file, 'w' )
    print( header_top, file=header, end='' )

for arg in args.argv:
    try:
        process_routine( arg )
    except Exception as ex:
        print( 'Error:' + arg + ':', ex )
        traceback.print_exc()
        print()
# end

if (args.header):
    print( header_bottom, file=header, end='' )
    header.close()
