#!/usr/bin/env python
#
# Attempts to find equations in Doxygen code and replace them with \f$ ... \f$.
# For instance:
#     /// solve A * X = B^H for matrices A, X, B.
# becomes
#     /// solve \f$ A X = B^H \f$ for matrices A, X, B.
# However, patterns are heuristic and incomplete, so make mistakes occur.
# Use with caution.

from __future__ import print_function

import sys
import re

def equation( line, subexpr=False ):
    if (subexpr):
        eqn = ''
    else:
        eqn = r'\f$ '
    pending = ''
    space = True
    while (line):
        # space
        s = re.search( r'^( +)(.*)', line )
        if (s):
            if (not space):
                eqn += s.group(1)
            line = s.group(2)
            space = True
            continue

        # variable or number
        s = re.search( r'^([A-Z]|VT|\d+)\b(.*)', line )
        if (s):
            eqn += s.group(1)
            line = s.group(2)
            space = False
            continue

        # binary operator * becomes implicit
        s = re.search( r'^(\*)(.*)', line )
        if (s):
            if (not space):
                eqn += ' '
            line = s.group(2)
            space = True
            continue

        # explicit binary operators + - /
        s = re.search( r'^([=\+\-\/])(.*)', line )
        if (s):
            eqn += s.group(1)
            line = s.group(2)
            space = False
            continue

        # exponents, transpose
        s = re.search( r'^(\^[HT2-9])\b(.*)', line )
        if (s):
            eqn += s.group(1)
            line = s.group(2)
            space = False
            continue

        # open parens
        s = re.search( r'^\((.*)', line )
        if (s):
            (expr, line) = equation( s.group(1), True )
            if (expr):
                eqn += '(' + expr + ')'
                space = False
            else:
                line = '(' + line
                break # done
            continue

        # close parens
        s = re.search( r'^\)', line )
        if (s):
            break  # finished subexpression; don't include ) in eqn

        # otherwise done
        break
    # end
    if (not subexpr):
        if (space):
            eqn += r'\f$ '
        else:
            eqn += r' \f$'
    return (eqn, line)
# end

def process( filename ):
    f_out = open( 'eqn/' + filename, 'w' )
    f_in  = open( filename )
    for line in f_in:
        s = re.search( r'^///', line )
        if (s):
            line = line.rstrip()
            ##if (re.search( r'^(.*?)\b((?:[A-Z]|VT) *(?:=|\^|\+|\-|\*|\/).*)', line )):
            ##    print( line, file=f_out )
            line2 = ''
            while (line):
                s = re.search( r'^(.*?)\b((?:[A-Z]|VT) *(?:=|\^|\+|\-|\*|\/).*)', line )
                if (s):
                    line2 += s.group(1)
                    (eqn, line) = equation( s.group(2) )
                    line2 += eqn
                else:
                    break
                # end
            # end
            print( line2 + line, file=f_out )
            ##print( file=f_out )
            ##end
        else:
            print( line, file=f_out, end='' )
        # end
    # end
# end

for arg in sys.argv[1:]:
    print( arg )
    process( arg )
