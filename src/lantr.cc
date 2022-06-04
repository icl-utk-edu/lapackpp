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
float lantr(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag, int64_t m, int64_t n,
    float const* A, int64_t lda )
{
    lapack_error_if( lda < m );

    // Safeguard m, n.
    if (uplo == Uplo::Lower)
        n = min( m, n );
    else
        m = min( m, n );

    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf ? m : 1);

    // allocate workspace
    lapack::vector< float > work( max( 1, lwork ) );

    return LAPACK_slantr(
        &norm_, &uplo_, &diag_, &m_, &n_,
        A, &lda_,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
double lantr(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag, int64_t m, int64_t n,
    double const* A, int64_t lda )
{
    lapack_error_if( lda < m );

    // Safeguard m, n.
    if (uplo == Uplo::Lower)
        n = min( m, n );
    else
        m = min( m, n );

    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf ? m : 1);

    // allocate workspace
    lapack::vector< double > work( max( 1, lwork ) );

    return LAPACK_dlantr(
        &norm_, &uplo_, &diag_, &m_, &n_,
        A, &lda_,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
float lantr(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag, int64_t m, int64_t n,
    std::complex<float> const* A, int64_t lda )
{
    lapack_error_if( lda < m );

    // Safeguard m, n.
    if (uplo == Uplo::Lower)
        n = min( m, n );
    else
        m = min( m, n );

    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf ? m : 1);

    // allocate workspace
    lapack::vector< float > work( max( 1, lwork ) );

    return LAPACK_clantr(
        &norm_, &uplo_, &diag_, &m_, &n_,
        (lapack_complex_float*) A, &lda_,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// Returns the value of the one norm, Frobenius norm,
/// infinity norm, or the element of largest absolute value of a
/// trapezoidal or triangular matrix A.
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
/// @param[in] uplo
///     Whether the matrix A is upper or lower trapezoidal.
///     - lapack::Uplo::Upper: Upper trapezoidal
///     - lapack::Uplo::Lower: Lower trapezoidal
///     - Note that A is triangular instead of trapezoidal if m = n.
///
/// @param[in] diag
///     Whether or not the matrix A has unit diagonal.
///     - lapack::Diag::NonUnit: Non-unit diagonal
///     - lapack::Diag::Unit: Unit diagonal
///
/// @param[in] m
///     The number of rows of the matrix A. m >= 0.
///     When m = 0, returns zero.
///     For upper, only the first min( m, n ) rows are accessed.
///     For lower, only the first min( m, n ) cols are accessed.
///     (LAPACK documentation says upper requires m <= n,
///     lower requires m >= n, but the code works for any m, n.
///     Other implementations may crash. LAPACK++ safeguards m, n to
///     work for any m, n.)
///
/// @param[in] n
///     The number of columns of the matrix A.
///     When n = 0, returns zero.
///
/// @param[in] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     The trapezoidal matrix A (A is triangular if m = n).
///     - If uplo = Upper, the leading m-by-n upper trapezoidal part of
///     the array A contains the upper trapezoidal matrix, and the
///     strictly lower triangular part of A is not referenced.
///
///     - If uplo = Lower, the leading m-by-n lower trapezoidal part of
///     the array A contains the lower trapezoidal matrix, and the
///     strictly upper triangular part of A is not referenced.
///
///     - Note that when diag = Unit, the diagonal elements of A are not
///     referenced and are assumed to be one.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(m,1).
///
/// @ingroup norm
double lantr(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag, int64_t m, int64_t n,
    std::complex<double> const* A, int64_t lda )
{
    lapack_error_if( lda < m );

    // Safeguard m, n.
    if (uplo == Uplo::Lower)
        n = min( m, n );
    else
        m = min( m, n );

    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf ? m : 1);

    // allocate workspace
    lapack::vector< double > work( max( 1, lwork ) );

    return LAPACK_zlantr(
        &norm_, &uplo_, &diag_, &m_, &n_,
        (lapack_complex_double*) A, &lda_,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
}

}  // namespace lapack
