// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"

// while [cz]syr are in LAPACK, [sd]syr are in BLAS,
// so we put them all in the blas namespace
namespace blas {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup syr
void syr(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *x, int64_t incx,
    std::complex<float>       *A, int64_t lda )
{
    // check arguments
    lapack_error_if( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    lapack_error_if( uplo != Uplo::Lower &&
               uplo != Uplo::Upper );
    lapack_error_if( n < 0 );
    lapack_error_if( lda < n );
    lapack_error_if( incx == 0 );

    lapack_int n_    = to_lapack_int( n );
    lapack_int lda_  = to_lapack_int( lda );
    lapack_int incx_ = to_lapack_int( incx );

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    char uplo_ = to_char( uplo );
    LAPACK_csyr( &uplo_, &n_,
                 (lapack_complex_float*) &alpha,
                 (lapack_complex_float*) x, &incx_,
                 (lapack_complex_float*) A, &lda_
    );
}

// -----------------------------------------------------------------------------
/// @ingroup syr
void syr(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *x, int64_t incx,
    std::complex<double>       *A, int64_t lda )
{
    // check arguments
    lapack_error_if( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    lapack_error_if( uplo != Uplo::Lower &&
               uplo != Uplo::Upper );
    lapack_error_if( n < 0 );
    lapack_error_if( lda < n );
    lapack_error_if( incx == 0 );

    lapack_int n_    = to_lapack_int( n );
    lapack_int lda_  = to_lapack_int( lda );
    lapack_int incx_ = to_lapack_int( incx );

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    char uplo_ = to_char( uplo );
    LAPACK_zsyr( &uplo_, &n_,
                 (lapack_complex_double*) &alpha,
                 (lapack_complex_double*) x, &incx_,
                 (lapack_complex_double*) A, &lda_
    );
}

}  // namespace blas
