#!/usr/bin/env python3
#
# Attempts to find equations in Doxygen code and replace them with $...$.
# For instance:
#     /// solve A * X = B^H for matrices A, X, B.
# becomes
#     /// solve $A X = B^H$ for matrices A, X, B.
# However, patterns are heuristic and incomplete, so mistakes occur.
# Use with caution.

from __future__ import print_function

import sys
import re
import os

debug = False

operators = {
    '<=': r'\le',
    '>=': r'\ge',
}

src = '../gen'
assert( os.path.exists( src ))

dst = '../eqn'
if (not os.path.exists( dst )):
    os.mkdir( dst )

def equation( line, subexpr=False ):
    if (debug):
        print( 'equation:', line + '; subexpr:', subexpr )

    if (subexpr):
        eqn = ''
    else:
        eqn = r'$'
    begin = True
    while (line):
        # no space after $ at beginning;
        # after that, add a space before most tokens (except ^T and punctuation)
        if (begin):
            begin = False
            space = ''
        else:
            space = ' '

        if (debug):
            print( 'eqn: <' + eqn + '> line: <' + line + '>' )

        # space
        s = re.search( r'^( +)(.*)', line )
        if (s):
            line = s.group(2)
            continue

        # variable
        s = re.search( r'^([a-zA-Z]|VT)(\d*)\b(.*)', line )
        if (s):
            eqn += space + s.group(1)
            if (s.group(2)):
                eqn += '_{' + s.group(2) + '}'
            line = s.group(3)
            continue

        # number
        s = re.search( r'^(\d+)\b(.*)', line )
        if (s):
            eqn += space + s.group(1)
            line = s.group(2)
            continue

        # binary operator * becomes implicit
        s = re.search( r'^(\*)(.*)', line )
        if (s):
            line = s.group(2)
            continue

        # explicit binary operators <= >=
        s = re.search( r'^(<=|>=)(.*)', line )
        if (s):
            eqn += space + operators[ s.group(1) ]
            line = s.group(2)
            continue

        # explicit binary operators = < > + - /
        s = re.search( r'^([=<>\+\-\/])(.*)', line )
        if (s):
            eqn += space + s.group(1)
            line = s.group(2)
            continue

        # exponents, transpose
        s = re.search( r'^(\^[HT2-9])\b(.*)', line )
        if (s):
            # don't print space
            eqn += s.group(1)
            line = s.group(2)
            continue

        # Greek, functions
        s = re.search( r'^(alpha|beta|lambda|Lambda|mu|sigma|Sigma|tau|sqrt|min|max)\b(.*)', line )
        if (s):
            eqn += space + '\\' + s.group(1)
            line = s.group(2)
            continue

        # capitalized Greek, functions
        s = re.search( r'^(MIN|MAX)\b(.*)', line )
        if (s):
            eqn += space + '\\' + s.group(1).lower()
            line = s.group(2)
            continue

        # norm
        # todo: use \lVert?
        s = re.search( r'^norm\((.*)', line )
        if (s):
            (expr, line) = equation( s.group(1), True )
            eqn += space + r'|| ' + expr + ' ||'
            continue

        # abs
        # todo: use \lvert?
        s = re.search( r'^abs\((.*)', line )
        if (s):
            (expr, line) = equation( s.group(1), True )
            eqn += space + r'| ' + expr + ' |'
            continue

        # inverse
        s = re.search( r'^inv\((.*)', line )
        if (s):
            (expr, line) = equation( s.group(1), True )
            if (re.search( '^\w+$', expr )):
                # safe to skip parens
                eqn += space + expr + '^{-1}'
            else:
                eqn += space + '(' + expr + ')^{-1}'
            continue

        # non-Latex functions
        s = re.search( r'^(diag)\((.*)', line )
        if (s):
            (expr, line) = equation( s.group(2), True )
            eqn += space + r'\; \text{' + s.group(1) + '}(' + expr + ') \; '
            continue

        # open parens
        s = re.search( r'^\((.*)', line )
        if (s):
            (expr, line) = equation( s.group(1), True )
            if (expr):
                eqn += space + '(' + expr + ')'
            else:
                # sub-expr didn't parse, so put ( back on line
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
            eqn += space + r'\dots'
            line = s.group(1)
            continue

        # punctuation
        ##s = re.search( r'^([.,;:])(.*)', line )
        s = re.search( r'^([,])(.*)', line )
        if (s):
            # don't print space
            eqn += s.group(1)
            line = s.group(2)
            continue

        # otherwise done
        break
    # end
    if (not subexpr):
        eqn += '$'
        # add space after $,
        # unless there's punctuation or already space such as \n in line.
        s = re.search( r'^[^.,;:\s]', line )
        if (s):
            eqn += ' '
    return (eqn, line)
# end

def process( arg ):
    filename = arg + '.cc'
    f_out = os.path.join( dst, filename )
    f_in  = os.path.join( src, filename )
    print( 'reading ' + f_in + '\nwriting ' + f_out )
    f_out = open( f_out, 'w' )
    f_in  = open( f_in )
    for line in f_in:
        s = re.search( r'^///', line )
        if (s):
            rest = line.rstrip()
            line = ''
            while (rest):
                #                                variable        Greek                                            operator          | functions
                s = re.search( r'^(.*?)\b((?: (?:[A-Z]\d* | VT | alpha | beta | lambda | mu | sigma | tau) \s* (?:=|\^|\+|\-|\*|\/) | norm\( | inv\( | diag\( | sqrt\( ).*)',
                               rest, re.X )
                if (s):
                    line += s.group(1)
                    (eqn, rest) = equation( s.group(2) )
                    if (debug):
                        print( 'eqn: ' + eqn + ', rest: ' + rest )
                    eqn = re.sub( r',\$( *)$', r'$,\1', eqn )  # move trailing punctuation (only commas) outside
                    line += eqn
                else:
                    break
                # end
            # end
            line += rest + '\n'
            # if the entire line is an indented equation, make it \[ ... \]
            # and put punctuation inside.
            line = re.sub( r'^///     \$(.*)\$([.,;:]?)$', r'///     \[ \1\2 \]', line )
        # end
        print( line, file=f_out, end='' )
    # end
# end

for arg in sys.argv[1:]:
    if (arg == '-d'):
        debug = True
    else:
        process( arg )
# end
