// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup ptsv_computational
int64_t pttrf(
    int64_t n,
    float* D,
    float* E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    LAPACK_spttrf(
        &n_,
        D,
        E, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ptsv_computational
int64_t pttrf(
    int64_t n,
    double* D,
    double* E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    LAPACK_dpttrf(
        &n_,
        D,
        E, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ptsv_computational
int64_t pttrf(
    int64_t n,
    float* D,
    std::complex<float>* E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    LAPACK_cpttrf(
        &n_,
        D,
        (lapack_complex_float*) E, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the $L D L^H$ factorization of a Hermitian
/// positive definite tridiagonal matrix A. The factorization may also
/// be regarded as having the form $A = U^H D U.$
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in,out] D
///     The vector D of length n.
///     On entry, the n diagonal elements of the tridiagonal matrix
///     A. On exit, the n diagonal elements of the diagonal matrix
///     D from the $L D L^H$ factorization of A.
///
/// @param[in,out] E
///     The vector E of length n-1.
///     On entry, the (n-1) subdiagonal elements of the tridiagonal
///     matrix A. On exit, the (n-1) subdiagonal elements of the
///     unit bidiagonal factor L from the $L D L^H$ factorization of A.
///     E can also be regarded as the superdiagonal of the unit
///     bidiagonal factor U from the $U^H D U$ factorization of A.
///
/// @return = 0: successful exit
/// @return > 0: if return value = i, the leading minor of order i is not
///     positive definite; if i < n, the factorization could not
///     be completed, while if i = n, the factorization was
///     completed, but D(n) <= 0.
///
/// @ingroup ptsv_computational
int64_t pttrf(
    int64_t n,
    double* D,
    std::complex<double>* E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    LAPACK_zpttrf(
        &n_,
        D,
        (lapack_complex_double*) E, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
