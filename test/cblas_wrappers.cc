// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Function definitions moved from cblas_wrappers.hh for ESSL compatability.

#include "lapack/fortran.h"
#include "cblas_wrappers.hh"

#include <complex>

// -----------------------------------------------------------------------------
void
cblas_symv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    int n,
    std::complex<float> alpha,
    std::complex<float> const* A, int lda,
    std::complex<float> const* x, int incx,
    std::complex<float> beta,
    std::complex<float>* yref, int incy )
{
    lapack_int n_    = lapack_int( n );
    lapack_int incx_ = lapack_int( incx );
    lapack_int incy_ = lapack_int( incy );
    lapack_int lda_  = lapack_int( lda );
    char uplo_ = lapack_uplo_const( uplo );
    if (layout == CblasRowMajor) {
        uplo_ = (uplo == CblasUpper ? 'l' : 'u');  // switch upper <=> lower
    }
    LAPACK_csymv(
        &uplo_, &n_,
        (lapack_complex_float*) &alpha,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) x, &incx_,
        (lapack_complex_float*) &beta,
        (lapack_complex_float*) yref, &incy_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}

// -----------------------------------------------------------------------------
void
cblas_symv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    int n,
    std::complex<double> alpha,
    std::complex<double> const* A, int lda,
    std::complex<double> const* x, int incx,
    std::complex<double> beta,
    std::complex<double>* yref, int incy )
{
    lapack_int n_    = lapack_int( n );
    lapack_int incx_ = lapack_int( incx );
    lapack_int incy_ = lapack_int( incy );
    lapack_int lda_  = lapack_int( lda );
    char uplo_ = lapack_uplo_const( uplo );
    if (layout == CblasRowMajor) {
        uplo_ = (uplo == CblasUpper ? 'l' : 'u');  // switch upper <=> lower
    }
    LAPACK_zsymv(
        &uplo_, &n_,
        (lapack_complex_double*) &alpha,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) x, &incx_,
        (lapack_complex_double*) &beta,
        (lapack_complex_double*) yref, &incy_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}

// -----------------------------------------------------------------------------
void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float>* A, int lda )
{
    lapack_int n_    = lapack_int( n );
    lapack_int incx_ = lapack_int( incx );
    lapack_int lda_  = lapack_int( lda );
    char uplo_ = lapack_uplo_const( uplo );
    if (layout == CblasRowMajor) {
        uplo_ = (uplo == CblasUpper ? 'l' : 'u');  // switch upper <=> lower
    }
    LAPACK_csyr(
        &uplo_, &n_,
        (lapack_complex_float*) &alpha,
        (lapack_complex_float*) x, &incx_,
        (lapack_complex_float*) A, &lda_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}

// -----------------------------------------------------------------------------
void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double>* A, int lda )
{
    lapack_int n_    = lapack_int( n );
    lapack_int incx_ = lapack_int( incx );
    lapack_int lda_  = lapack_int( lda );
    char uplo_ = lapack_uplo_const( uplo );
    if (layout == CblasRowMajor) {
        uplo_ = (uplo == CblasUpper ? 'l' : 'u');  // switch upper <=> lower
    }
    LAPACK_zsyr(
        &uplo_, &n_,
        (lapack_complex_double*) &alpha,
        (lapack_complex_double*) x, &incx_,
        (lapack_complex_double*) A, &lda_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}
