// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup norm
float lanhb(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<float> const* AB, int64_t ldab )
{
    char norm_ = to_char( norm );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int ldab_ = to_lapack_int( ldab );

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    lapack::vector< float > work( max(1,lwork) );

    return LAPACK_clanhb(
        &norm_, &uplo_, &n_, &kd_,
        (lapack_complex_float*) AB, &ldab_,
        &work[0]
    );
}

// -----------------------------------------------------------------------------
/// Returns the value of the one norm, Frobenius norm,
/// infinity norm, or the element of largest absolute value of an
/// n-by-n hermitian band matrix A, with kd super-diagonals.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this is an alias for `lapack::lansb`.
/// For complex symmetric matrices, see `lapack::lansb`.
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
///     - lapack::Uplo::Upper: Upper triangular
///     - lapack::Uplo::Lower: Lower triangular
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
///     The upper or lower triangle of the hermitian band matrix A,
///     stored in the first kd+1 rows of AB. The j-th column of A is
///     stored in the j-th column of the array AB as follows:
///     - if uplo = Upper, AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd) <= i <= j;
///     - if uplo = Lower, AB(1+i-j,j) = A(i,j) for j <= i <= min(n,j+kd).
///     - Note that the imaginary parts of the diagonal elements need
///     not be set and are assumed to be zero.
///
/// @param[in] ldab
///     The leading dimension of the array AB. ldab >= kd+1.
///
/// @ingroup norm
double lanhb(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<double> const* AB, int64_t ldab )
{
    char norm_ = to_char( norm );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int ldab_ = to_lapack_int( ldab );

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    lapack::vector< double > work( max(1,lwork) );

    return LAPACK_zlanhb(
        &norm_, &uplo_, &n_, &kd_,
        (lapack_complex_double*) AB, &ldab_,
        &work[0]
    );
}

}  // namespace lapack
