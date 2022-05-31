// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//
// -----------------------------------------------------------------------------

#include "lapack/fortran.h"
#include "lapack.hh"

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

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( n              > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( lda            > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(incx) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(incy) > std::numeric_limits<lapack_int>::max() );
    }

    lapack_int n_    = (lapack_int) n;
    lapack_int lda_  = (lapack_int) lda;
    lapack_int incx_ = (lapack_int) incx;
    lapack_int incy_ = (lapack_int) incy;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    char uplo_ = uplo2char( uplo );
    LAPACK_csymv( &uplo_, &n_,
                (lapack_complex_float*) &alpha,
                (lapack_complex_float*) A, &lda_,
                (lapack_complex_float*) x, &incx_,
                (lapack_complex_float*) &beta,
                (lapack_complex_float*) y, &incy_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
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

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( n              > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( lda            > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(incx) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(incy) > std::numeric_limits<lapack_int>::max() );
    }

    lapack_int n_    = (lapack_int) n;
    lapack_int lda_  = (lapack_int) lda;
    lapack_int incx_ = (lapack_int) incx;
    lapack_int incy_ = (lapack_int) incy;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    char uplo_ = uplo2char( uplo );
    LAPACK_zsymv( &uplo_, &n_,
                (lapack_complex_double*) &alpha,
                (lapack_complex_double*) A, &lda_,
                (lapack_complex_double*) x, &incx_,
                (lapack_complex_double*) &beta,
                (lapack_complex_double*) y, &incy_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}

}  // namespace blas

