#!/usr/bin/env python3
#
# Usage: ./gen.py basenames
# where basenames are like getrf, gesv, posv, ...
#
# Generates wrapper in ../gen, copies to ../src, and attempts to compile.

from __future__ import print_function

import sys
import os

pwd = os.getcwd()

def run( cmd ):
    print( cmd )
    os.system( cmd )
# end

for arg in sys.argv[1:]:
    try:
        gen = '../gen/' + arg + '.cc'
        src = '../src/' + arg + '.cc'
        if (os.path.exists( src )):
            print( src + ' already exists; skipping' )
            continue
        run( './wrapper_gen.py ' + arg )
        run( 'cp ' + gen + ' ' + src )
        os.chdir( '../src' )
        run( 'make ' + arg + '.o' )
    except:
        pass

    os.chdir( pwd )
# end
