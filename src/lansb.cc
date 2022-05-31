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
float lansb(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n, int64_t kd,
    float const* AB, int64_t ldab )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int ldab_ = (lapack_int) ldab;

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    lapack::vector< float > work( max(1,lwork) );

    return LAPACK_slansb(
        &norm_, &uplo_, &n_, &kd_,
        AB, &ldab_,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
double lansb(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n, int64_t kd,
    double const* AB, int64_t ldab )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int ldab_ = (lapack_int) ldab;

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    lapack::vector< double > work( max(1,lwork) );

    return LAPACK_dlansb(
        &norm_, &uplo_, &n_, &kd_,
        AB, &ldab_,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
float lansb(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<float> const* AB, int64_t ldab )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int ldab_ = (lapack_int) ldab;

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    lapack::vector< float > work( max(1,lwork) );

    return LAPACK_clansb(
        &norm_, &uplo_, &n_, &kd_,
        (lapack_complex_float*) AB, &ldab_,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// Returns the value of the one norm, Frobenius norm,
/// infinity norm, or the element of largest absolute value of an
/// n-by-n symmetric band matrix A, with kd super-diagonals.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, `lapack::lanhb` is an alias for this.
/// For complex Hermitian matrices, see `lapack::lanhb`.
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
///     Whether the upper or lower triangular part of the
///     band matrix A is supplied.
///     - lapack::Uplo::Upper: Upper triangular part is supplied
///     - lapack::Uplo::Lower: Lower triangular part is supplied
///
/// @param[in] n
///     The order of the matrix A. n >= 0. When n = 0, returns zero.
///
/// @param[in] kd
///     The number of super-diagonals or sub-diagonals of the
///     band matrix A. kd >= 0.
///
/// @param[in] AB
///     The kd+1-by-n matrix AB, stored in an ldab-by-n array.
///     The upper or lower triangle of the symmetric band matrix A,
///     stored in the first kd+1 rows of AB. The j-th column of A is
///     stored in the j-th column of the array AB as follows:
///     - if uplo = Upper, AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd) <= i <= j;
///     - if uplo = Lower, AB(1+i-j,j) = A(i,j) for j <= i <= min(n,j+kd).
///
/// @param[in] ldab
///     The leading dimension of the array AB. ldab >= kd+1.
///
/// @ingroup norm
double lansb(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<double> const* AB, int64_t ldab )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int ldab_ = (lapack_int) ldab;

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    lapack::vector< double > work( max(1,lwork) );

    return LAPACK_zlansb(
        &norm_, &uplo_, &n_, &kd_,
        (lapack_complex_double*) AB, &ldab_,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
}

}  // namespace lapack
