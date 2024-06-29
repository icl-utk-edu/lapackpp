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
float lanhp(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    std::complex<float> const* AP )
{
    char norm_ = to_char( norm );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    lapack::vector< float > work( max(1,lwork) );

    return LAPACK_clanhp(
        &norm_, &uplo_, &n_,
        (lapack_complex_float*) AP,
        &work[0]
    );
}

// -----------------------------------------------------------------------------
/// Returns the value of the one norm, Frobenius norm,
/// infinity norm, or the element of largest absolute value of a
/// complex hermitian matrix A, supplied in packed form.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this is an alias for `lapack::lansp`.
/// For complex symmetric matrices, see `lapack::lansp`.
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
///     hermitian matrix A is supplied.
///     - lapack::Uplo::Upper: Upper triangular part of A is supplied
///     - lapack::Uplo::Lower: Lower triangular part of A is supplied
///
/// @param[in] n
///     The order of the matrix A. n >= 0. When n = 0, returns zero.
///
/// @param[in] AP
///     The vector AP of length n*(n+1)/2.
///     The upper or lower triangle of the hermitian matrix A, packed
///     columnwise in a linear array. The j-th column of A is stored
///     in the array AP as follows:
///     - if uplo = Upper, AP(i + (j-1)*j/2) = A(i,j) for 1 <= i <= j;
///     - if uplo = Lower, AP(i + (j-1)*(2n-j)/2) = A(i,j) for j <= i <= n.
///     - Note that the imaginary parts of the diagonal elements need
///     not be set and are assumed to be zero.
///
/// @ingroup norm
double lanhp(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    std::complex<double> const* AP )
{
    char norm_ = to_char( norm );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    lapack::vector< double > work( max(1,lwork) );

    return LAPACK_zlanhp(
        &norm_, &uplo_, &n_,
        (lapack_complex_double*) AP,
        &work[0]
    );
}

}  // namespace lapack
