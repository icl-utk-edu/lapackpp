#!/usr/bin/env python3
#
# Usage: ./regen.py
#
# For all src/*.o files, regenerate wrappers in ../gen/*.cc
# Does NOT touch ../src/*.cc

from __future__ import print_function

import sys
import os
import re

def run( cmd ):
    print( cmd )
    os.system( cmd )
# end

files = os.listdir( '../src' )
for f in files:
    s = re.search( r'^(\w+)\.cc', f )
    if (s):
        arg = s.group(1)
        run( './wrapper_gen.py ' + arg )
# end
