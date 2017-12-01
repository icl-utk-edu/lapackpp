#!/bin/sh

export   square='--dim 100:500:100'
export     tall='--dim 200:1000:200x100:500:100'
export     wide='--dim 100:500:100x200:1000:200'
export       mn="${square} ${tall} ${wide}"
export      mnk="${mn} --dim 100x300x600,300x100x600,100x600x300,300x600x100,600x100x300,600x300x100"

export  nk_tall='--dim 1x200:1000:200x100:500:100'
export  nk_wide='--dim 1x100:500:100x200:1000:200'
export       nk="${square} ${nk_tall} ${nk_wide}"

export incx_pos='--incx 1,2'
export     incx='--incx 1,2,-1,-2'
export incy_pos='--incy 1,2'
export     incy='--incy 1,2,-1,-2'

export     type='--type=s,d,c,z'
export    trans='--trans=n,t,c'
export     uplo='--uplo=l,u'
export     norm='--norm 1,inf,fro,max'
export     diag='--diag n,u'
export   direct='--direct f,b'
export   storev='--storev c,r'
export     side='--side l,r'

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

# symmetric indefinite
./test sysv  ${type} ${square} ${uplo}
./test sytrf ${type} ${square} ${uplo}
./test sytrs ${type} ${square} ${uplo}
./test sytri ${type} ${square} ${uplo}
./test sycon ${type} ${square} ${uplo}
./test syrfs ${type} ${square} ${uplo}

# auxilary
./test lacpy ${type} ${mn} --matrixtype=g,l,u
./test laset ${type} ${mn} --matrixtype=g,l,u
./test laswp ${type} ${mn}

# auxilary - householder
./test larfg ${type} ${square} ${incx_pos}
./test larf  ${type} ${mn}     ${incx} ${side}
./test larfx ${type} ${mn}     ${side}
./test larfb ${type} ${mnk}    ${side} ${trans} ${direct} ${storev}
./test larft ${type} ${nk}     ${direct} ${storev}

# auxilary - norms
./test lange ${type} ${mn}     ${norm}
./test lanhe ${type} ${square} ${norm} ${uplo}
./test lansy ${type} ${square} ${norm} ${uplo}
./test lantr ${type} ${square} ${norm} ${uplo} ${diag}

# additional blas
./test syr   ${type} ${square} ${uplo}
