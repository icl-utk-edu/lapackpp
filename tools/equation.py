#!/usr/bin/env python
#
# Attempts to find equations in Doxygen code and replace them with \f$ ... \f$.
# For instance:
#     /// solve A * X = B^H for matrices A, X, B.
# becomes
#     /// solve \f$ A X = B^H \f$ for matrices A, X, B.
# However, patterns are heuristic and incomplete, so mistakes occur.
# Use with caution.

from __future__ import print_function

import sys
import re

debug = False

def equation( line, subexpr=False ):
    if (debug):
        print( 'equation:', line, 'subexpr', subexpr )

    if (subexpr):
        eqn = ''
    else:
        eqn = r'\f$ '
    space = True
    while (line):
        if (debug):
            print( 'eqn: <' + eqn + '> line: <' + line + '>' )

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

        # Greek, functions
        s = re.search( r'^(alpha|beta|lambda|Lambda|sigma|Sigma|tau|sqrt|min|max)\b(.*)', line )
        if (s):
            eqn += '\\' + s.group(1)
            line = s.group(2)
            space = False
            continue

        # capitalized Greek, functions
        s = re.search( r'^(MIN|MAX)\b(.*)', line )
        if (s):
            eqn += '\\' + s.group(1).lower()
            line = s.group(2)
            space = False
            continue

        # norm
        s = re.search( r'^norm\((.*)', line )
        if (s):
            (expr, line) = equation( s.group(1), True )
            eqn += r'|| ' + expr + ' ||'
            continue

        # abs
        s = re.search( r'^abs\((.*)', line )
        if (s):
            (expr, line) = equation( s.group(1), True )
            eqn += r'| ' + expr + ' |'
            continue

        # inverse
        s = re.search( r'^inv\((.*)', line )
        if (s):
            (expr, line) = equation( s.group(1), True )
            if (re.search( '^\w+$', expr )):
                # safe to skip parens
                eqn += expr + '^{-1}'
            else:
                eqn += '(' + expr + ')^{-1}'
            space = True
            continue

        # non-Latex functions
        s = re.search( r'^(diag)\((.*)', line )
        if (s):
            (expr, line) = equation( s.group(2), True )
            if (not space): eqn += ' '
            eqn += r'\; \text{' + s.group(1) + '}(' + expr + ') \; '
            space = True
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
        if (subexpr):
            s = re.search( r'^\)(.*)', line )
            if (s):
                # finished subexpression; don't include ) in eqn
                line = s.group(1)
                if (debug):
                    print( 'finish subexpr:', eqn, '\nline:', line )
                break

        # ellipsis
        s = re.search( r'^\. ?\. ?\.(.*)', line )
        if (s):
            if (not space):
                eqn += ' '
            eqn += r'\dots '
            space = True
            line = s.group(1)
            continue

        # punctuation
        s = re.search( r'^([.,;:])(.*)', line )
        if (s):
            eqn += s.group(1)
            line = s.group(2)
            space = False
            continue

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

def process( arg ):
    filename = arg + '.cc'
    f_out = open( '../eqn/' + filename, 'w' )
    f_in  = open( '../gen/' + filename )
    for line in f_in:
        s = re.search( r'^///', line )
        if (s):
            line = line.rstrip()
            ##if (re.search( r'^(.*?)\b((?:[A-Z]|VT) *(?:=|\^|\+|\-|\*|\/).*)', line )):
            ##    print( line, file=f_out )
            line2 = ''
            while (line):
                s = re.search( r'^(.*?)\b((?:(?:[A-Z]|VT) *(?:=|\^|\+|\-|\*|\/)|norm\(|inv\(|diag\(|sqrt\(|alpha|beta|lambda|sigma|tau).*)', line )
                if (s):
                    line2 += s.group(1)
                    (eqn, line) = equation( s.group(2) )
                    eqn = re.sub( r'(\\f\$)([.,;:])', r'\2\1', eqn )
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
    if (arg == '-d'):
        debug = True
    else:
        print( arg )
        process( arg )
# end
