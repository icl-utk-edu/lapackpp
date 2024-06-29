// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//
// -----------------------------------------------------------------------------

#include "lapack/fortran.h"
#include "lapack.hh"
#include "lapack_internal.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for c, z precisions.


/// @ingroup symv
void symv(
        blas::Layout layout,
        blas::Uplo uplo,
        int64_t n,
        std::complex<float> alpha,
        std::complex<float> const *A, int64_t lda,
        std::complex<float> const *x, int64_t incx,
        std::complex<float> beta,
        std::complex<float>       *y, int64_t incy )
{
    // check arguments
    lapack_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    lapack_error_if( uplo != Uplo::Upper &&
                   uplo != Uplo::Lower );
    lapack_error_if( n < 0 );
    lapack_error_if( lda < n );
    lapack_error_if( incx == 0 );
    lapack_error_if( incy == 0 );

    lapack_int n_    = to_lapack_int( n );
    lapack_int lda_  = to_lapack_int( lda );
    lapack_int incx_ = to_lapack_int( incx );
    lapack_int incy_ = to_lapack_int( incy );

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    char uplo_ = to_char( uplo );
    LAPACK_csymv( &uplo_, &n_,
                (lapack_complex_float*) &alpha,
                (lapack_complex_float*) A, &lda_,
                (lapack_complex_float*) x, &incx_,
                (lapack_complex_float*) &beta,
                (lapack_complex_float*) y, &incy_
    );
}

// -----------------------------------------------------------------------------
/// @ingroup symv
void symv(
        blas::Layout layout,
        blas::Uplo uplo,
        int64_t n,
        std::complex<double> alpha,
        std::complex<double> const *A, int64_t lda,
        std::complex<double> const *x, int64_t incx,
        std::complex<double> beta,
        std::complex<double>       *y, int64_t incy )
{
    // check arguments
    lapack_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    lapack_error_if( uplo != Uplo::Upper &&
                   uplo != Uplo::Lower );
    lapack_error_if( n < 0 );
    lapack_error_if( lda < n );
    lapack_error_if( incx == 0 );
    lapack_error_if( incy == 0 );

    lapack_int n_    = to_lapack_int( n );
    lapack_int lda_  = to_lapack_int( lda );
    lapack_int incx_ = to_lapack_int( incx );
    lapack_int incy_ = to_lapack_int( incy );

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    char uplo_ = to_char( uplo );
    LAPACK_zsymv( &uplo_, &n_,
                (lapack_complex_double*) &alpha,
                (lapack_complex_double*) A, &lda_,
                (lapack_complex_double*) x, &incx_,
                (lapack_complex_double*) &beta,
                (lapack_complex_double*) y, &incy_
    );
}

}  // namespace blas

