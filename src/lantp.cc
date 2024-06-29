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
float lantp(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag, int64_t n,
    float const* AP )
{
    char norm_ = to_char( norm );
    char uplo_ = to_char( uplo );
    char diag_ = to_char( diag );
    lapack_int n_ = to_lapack_int( n );

    // from docs
    int64_t lwork = (norm == Norm::Inf ? n : 1);

    // allocate workspace
    lapack::vector< float > work( max(1,lwork) );

    return LAPACK_slantp(
        &norm_, &uplo_, &diag_, &n_,
        AP,
        &work[0]
    );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
double lantp(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag, int64_t n,
    double const* AP )
{
    char norm_ = to_char( norm );
    char uplo_ = to_char( uplo );
    char diag_ = to_char( diag );
    lapack_int n_ = to_lapack_int( n );

    // from docs
    int64_t lwork = (norm == Norm::Inf ? n : 1);

    // allocate workspace
    lapack::vector< double > work( max(1,lwork) );

    return LAPACK_dlantp(
        &norm_, &uplo_, &diag_, &n_,
        AP,
        &work[0]
    );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
float lantp(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag, int64_t n,
    std::complex<float> const* AP )
{
    char norm_ = to_char( norm );
    char uplo_ = to_char( uplo );
    char diag_ = to_char( diag );
    lapack_int n_ = to_lapack_int( n );

    // from docs
    int64_t lwork = (norm == Norm::Inf ? n : 1);

    // allocate workspace
    lapack::vector< float > work( max(1,lwork) );

    return LAPACK_clantp(
        &norm_, &uplo_, &diag_, &n_,
        (lapack_complex_float*) AP,
        &work[0]
    );
}

// -----------------------------------------------------------------------------
/// Returns the value of the one norm, Frobenius norm,
/// infinity norm, or the element of largest absolute value of a
/// triangular matrix A, supplied in packed form.
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
///     Whether the matrix A is upper or lower triangular.
///     - lapack::Uplo::Upper: Upper triangular
///     - lapack::Uplo::Lower: Lower triangular
///
/// @param[in] diag
///     Whether or not the matrix A is unit triangular.
///     - lapack::Diag::NonUnit: Non-unit triangular
///     - lapack::Diag::Unit: Unit triangular
///
/// @param[in] n
///     The order of the matrix A. n >= 0. When n = 0, returns zero.
///
/// @param[in] AP
///     The n-by-n matrix AP, packed in an (n*(n+1)/2) array.
///     The upper or lower triangular matrix A, packed columnwise in
///     a linear array. The j-th column of A is stored in the array
///     AP as follows:
///     - if uplo = Upper, AP(i + (j-1)*j/2) = A(i,j) for 1 <= i <= j;
///     - if uplo = Lower, AP(i + (j-1)*(2n-j)/2) = A(i,j) for j <= i <= n.
///     - Note that when diag = Unit, the elements of the array AP
///     corresponding to the diagonal elements of the matrix A are
///     not referenced, but are assumed to be one.
///
/// @ingroup norm
double lantp(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag, int64_t n,
    std::complex<double> const* AP )
{
    char norm_ = to_char( norm );
    char uplo_ = to_char( uplo );
    char diag_ = to_char( diag );
    lapack_int n_ = to_lapack_int( n );

    // from docs
    int64_t lwork = (norm == Norm::Inf ? n : 1);

    // allocate workspace
    lapack::vector< double > work( max(1,lwork) );

    return LAPACK_zlantp(
        &norm_, &uplo_, &diag_, &n_,
        (lapack_complex_double*) AP,
        &work[0]
    );
}

}  // namespace lapack
