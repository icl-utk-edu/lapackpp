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
float lansp(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    float const* AP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    lapack::vector< float > work( max(1,lwork) );

    return LAPACK_slansp(
        &norm_, &uplo_, &n_,
        AP,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
double lansp(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    double const* AP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    lapack::vector< double > work( max(1,lwork) );

    return LAPACK_dlansp(
        &norm_, &uplo_, &n_,
        AP,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
float lansp(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    std::complex<float> const* AP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    lapack::vector< float > work( max(1,lwork) );

    return LAPACK_clansp(
        &norm_, &uplo_, &n_,
        (lapack_complex_float*) AP,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// Returns the value of the one norm, Frobenius norm,
/// infinity norm, or the element of largest absolute value of a
/// complex symmetric matrix A, supplied in packed form.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, `lapack::lanhp` is an alias for this.
/// For complex Hermitian matrices, see `lapack::lanhp`.
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
///     symmetric matrix A is supplied.
///     - lapack::Uplo::Upper: Upper triangular part of A is supplied
///     - lapack::Uplo::Lower: Lower triangular part of A is supplied
///
/// @param[in] n
///     The order of the matrix A. n >= 0. When n = 0, returns zero.
///
/// @param[in] AP
///     The vector AP of length n*(n+1)/2.
///     The upper or lower triangle of the symmetric matrix A, packed
///     columnwise in a linear array. The j-th column of A is stored
///     in the array AP as follows:
///     - if uplo = Upper, AP(i + (j-1)*j/2) = A(i,j) for 1 <= i <= j;
///     - if uplo = Lower, AP(i + (j-1)*(2n-j)/2) = A(i,j) for j <= i <= n.
///
/// @ingroup norm
double lansp(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    std::complex<double> const* AP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    lapack::vector< double > work( max(1,lwork) );

    return LAPACK_zlansp(
        &norm_, &uplo_, &n_,
        (lapack_complex_double*) AP,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
}

}  // namespace lapack
