// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup norm
float lange(
    lapack::Norm norm, int64_t m, int64_t n,
    float const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf ? m : 1);

    // allocate workspace
    lapack::vector< float > work( max( 1, lwork ) );

    return LAPACK_slange(
        &norm_, &m_, &n_,
        A, &lda_,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
double lange(
    lapack::Norm norm, int64_t m, int64_t n,
    double const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf ? m : 1);

    // allocate workspace
    lapack::vector< double > work( max( 1, lwork ) );

    return LAPACK_dlange(
        &norm_, &m_, &n_,
        A, &lda_,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
float lange(
    lapack::Norm norm, int64_t m, int64_t n,
    std::complex<float> const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf ? m : 1);

    // allocate workspace
    lapack::vector< float > work( max( 1, lwork ) );

    return LAPACK_clange(
        &norm_, &m_, &n_,
        (lapack_complex_float*) A, &lda_,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// Returns the value of the one norm, Frobenius norm,
/// infinity norm, or the element of largest absolute value of a
/// complex matrix A.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] norm
///     The value to be returned:
///     - lapack::Norm::Max: max norm: max(abs(A(i,j))).
///                          Note this is not a consistent matrix norm.
///     - lapack::Norm::One: one norm: maximum column sum
///     - lapack::Norm::Inf: infinity norm: maximum row sum
///     - lapack::Norm::Fro: Frobenius norm: square root of sum of squares
///
/// @param[in] m
///     The number of rows of the matrix A. m >= 0. When m = 0, returns zero.
///
/// @param[in] n
///     The number of columns of the matrix A. n >= 0. When n = 0, returns zero.
///
/// @param[in] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(m,1).
///
/// @ingroup norm
double lange(
    lapack::Norm norm, int64_t m, int64_t n,
    std::complex<double> const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf ? m : 1);

    // allocate workspace
    lapack::vector< double > work( max( 1, lwork ) );

    return LAPACK_zlange(
        &norm_, &m_, &n_,
        (lapack_complex_double*) A, &lda_,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}

}  // namespace lapack
