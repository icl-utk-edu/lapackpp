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
float langb(
    lapack::Norm norm, int64_t n, int64_t kl, int64_t ku,
    float const* AB, int64_t ldab )
{
    char norm_ = to_char( norm );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int ldab_ = to_lapack_int( ldab );

    // from docs
    int64_t lwork = (norm == Norm::Inf ? n : 1);

    // allocate workspace
    lapack::vector< float > work( max(1,lwork) );

    return LAPACK_slangb(
        &norm_, &n_, &kl_, &ku_,
        AB, &ldab_,
        &work[0]
    );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
double langb(
    lapack::Norm norm, int64_t n, int64_t kl, int64_t ku,
    double const* AB, int64_t ldab )
{
    char norm_ = to_char( norm );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int ldab_ = to_lapack_int( ldab );

    // from docs
    int64_t lwork = (norm == Norm::Inf ? n : 1);

    // allocate workspace
    lapack::vector< double > work( max(1,lwork) );

    return LAPACK_dlangb(
        &norm_, &n_, &kl_, &ku_,
        AB, &ldab_,
        &work[0]
    );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
float langb(
    lapack::Norm norm, int64_t n, int64_t kl, int64_t ku,
    std::complex<float> const* AB, int64_t ldab )
{
    char norm_ = to_char( norm );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int ldab_ = to_lapack_int( ldab );

    // from docs
    int64_t lwork = (norm == Norm::Inf ? n : 1);

    // allocate workspace
    lapack::vector< float > work( max(1,lwork) );

    return LAPACK_clangb(
        &norm_, &n_, &kl_, &ku_,
        (lapack_complex_float*) AB, &ldab_,
        &work[0]
    );
}

// -----------------------------------------------------------------------------
/// Returns the value of the one norm, Frobenius norm,
/// infinity norm, or the element of largest absolute value of an
/// n-by-n band matrix A, with kl sub-diagonals and ku super-diagonals.
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
/// @param[in] n
///     The order of the matrix A. n >= 0. When n = 0, returns zero.
///
/// @param[in] kl
///     The number of sub-diagonals of the matrix A. kl >= 0.
///
/// @param[in] ku
///     The number of super-diagonals of the matrix A. ku >= 0.
///
/// @param[in] AB
///     The n-by-n band matrix AB, stored in an ldab-by-n array.
///     The band matrix A, stored in rows 1 to kl+ku+1. The j-th
///     column of A is stored in the j-th column of the array AB as
///     follows:
///     AB(ku+1+i-j,j) = A(i,j) for max(1,j-ku) <= i <= min(n,j+kl).
///
/// @param[in] ldab
///     The leading dimension of the array AB. ldab >= kl+ku+1.
///
/// @ingroup norm
double langb(
    lapack::Norm norm, int64_t n, int64_t kl, int64_t ku,
    std::complex<double> const* AB, int64_t ldab )
{
    char norm_ = to_char( norm );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int ldab_ = to_lapack_int( ldab );

    // from docs
    int64_t lwork = (norm == Norm::Inf ? n : 1);

    // allocate workspace
    lapack::vector< double > work( max(1,lwork) );

    return LAPACK_zlangb(
        &norm_, &n_, &kl_, &ku_,
        (lapack_complex_double*) AB, &ldab_,
        &work[0]
    );
}

}  // namespace lapack
