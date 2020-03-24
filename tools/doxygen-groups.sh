#!/bin/sh
#
# Finds doxygen groups that are in use,  sorts & puts in file "ingroup"
# Finds doxygen groups that are defined, sorts & puts in file "defgroup"
# Doing
#     diff ingroup defgroup
# provides an easy way to see what groups are used vs. defined.

find include src test \
    -name '*.h' -o -name '*.hh' -o -name '*.c' -o -name '*.cc' \
    > src.txt

egrep -h '@(addto|in)group' `cat src.txt` |
	perl -pe 's#/// +##;  s/^ *\*//;  s/^ +//;  s/\@(addto|in)group/\@group/;' | \
	sort --unique > ingroup.txt

egrep -h '^ *@defgroup' docs/doxygen/groups.dox | \
    egrep -v 'group_' | \
    perl -pe 's/^ *\@defgroup +(\w+).*/\@group $1/;' | \
	sort > defgroup.txt

echo "Undefined groups (missing in docs/doxygen/groups.dox):"
diff ingroup.txt defgroup.txt | grep '^<'
echo

echo "Unused groups (unused in src, test):"
diff ingroup.txt defgroup.txt | grep '^>'
echo

#opendiff ingroup.txt defgroup.txt
