#!/bin/sh

export   square='--dim 100:500:100'
export     tall='--dim 200:1000:200x100:500:100'
export     wide='--dim 100:500:100x200:1000:200'
export       mn="${square} ${tall} ${wide}"
export     type='--type=s,d,c,z'
export    trans='--trans=n,t,c'
export     uplo='--uplo=l,u'

# echo commands
set -x

# LU
./test gesv  ${type} ${square}
./test getrf ${type} ${mn}
./test getrs ${type} ${square} ${trans}
./test getri ${type} ${square}
./test gecon ${type} ${square}
./test gerfs ${type} ${square} ${trans}
./test geequ ${type} ${square}

# Cholesky
./test posv  ${type} ${square} ${uplo}
./test potrf ${type} ${square} ${uplo}
./test potrs ${type} ${square} ${uplo}
./test potri ${type} ${square} ${uplo}
./test pocon ${type} ${square} ${uplo}
./test porfs ${type} ${square} ${uplo}
./test poequ ${type} ${square} # only diagonal elements

# auxilary - norms
./test lange ${type} ${mn}
./test lanhe ${type} ${square} ${uplo}
./test lansy ${type} ${square} ${uplo}
./test lantr ${type} ${square} ${uplo}
