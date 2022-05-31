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
/// @ingroup norm
float langt(
    lapack::Norm norm, int64_t n,
    float const* DL,
    float const* D,
    float const* DU )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    lapack_int n_ = (lapack_int) n;

    return LAPACK_slangt(
        &norm_, &n_,
        DL,
        D,
        DU
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
double langt(
    lapack::Norm norm, int64_t n,
    double const* DL,
    double const* D,
    double const* DU )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    lapack_int n_ = (lapack_int) n;

    return LAPACK_dlangt(
        &norm_, &n_,
        DL,
        D,
        DU
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
float langt(
    lapack::Norm norm, int64_t n,
    std::complex<float> const* DL,
    std::complex<float> const* D,
    std::complex<float> const* DU )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    lapack_int n_ = (lapack_int) n;

    return LAPACK_clangt(
        &norm_, &n_,
        (lapack_complex_float*) DL,
        (lapack_complex_float*) D,
        (lapack_complex_float*) DU
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// Returns the value of the one norm, Frobenius norm,
/// infinity norm, or the element of largest absolute value of a
/// tridiagonal matrix A.
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
/// @param[in] DL
///     The vector DL of length n-1.
///     The (n-1) sub-diagonal elements of A.
///
/// @param[in] D
///     The vector D of length n.
///     The diagonal elements of A.
///
/// @param[in] DU
///     The vector DU of length n-1.
///     The (n-1) super-diagonal elements of A.
///
/// @ingroup norm
double langt(
    lapack::Norm norm, int64_t n,
    std::complex<double> const* DL,
    std::complex<double> const* D,
    std::complex<double> const* DU )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    lapack_int n_ = (lapack_int) n;

    return LAPACK_zlangt(
        &norm_, &n_,
        (lapack_complex_double*) DL,
        (lapack_complex_double*) D,
        (lapack_complex_double*) DU
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}

}  // namespace lapack
