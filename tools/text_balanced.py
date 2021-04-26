# Extracts initial substring delimited by characters start and end.
# Returns substring, excluding delimiters, and remainder. Example:
# (extracted, remainder) = extract_bracketed( "(n*(n+1)) other", '(', ')' )
# extracted = "n*(n+1)"
# remainder = " other"
# see https://stackoverflow.com/questions/1651487/python-parsing-bracketed-blocks
def extract_bracketed( txt, start, end ):
    if (txt[0] != start):
        return (None, txt)
    cnt = 0
    for i in range( 0, len( txt )):
        if (txt[i] == start):
            cnt += 1
        elif (txt[i] == end):
            cnt -= 1
        if (cnt == 0):
            return (txt[1:i], txt[i+1:])
    # unbalanced
    return (None, txt)
# end
