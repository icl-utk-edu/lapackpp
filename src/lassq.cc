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
/// @ingroup auxiliary
void lassq(
    int64_t n,
    float const* x, int64_t incx,
    float* scale,
    float* sumsq )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(incx) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int incx_ = (lapack_int) incx;

    LAPACK_slassq(
        &n_,
        x, &incx_, scale, sumsq );
}

// -----------------------------------------------------------------------------
/// @ingroup auxiliary
void lassq(
    int64_t n,
    double const* x, int64_t incx,
    double* scale,
    double* sumsq )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(incx) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int incx_ = (lapack_int) incx;

    LAPACK_dlassq(
        &n_,
        x, &incx_, scale, sumsq );
}

// -----------------------------------------------------------------------------
/// @ingroup auxiliary
void lassq(
    int64_t n,
    std::complex<float> const* x, int64_t incx,
    float* scale,
    float* sumsq )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(incx) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int incx_ = (lapack_int) incx;

    LAPACK_classq(
        &n_,
        (lapack_complex_float*) x, &incx_, scale, sumsq );
}

// -----------------------------------------------------------------------------
/// Compute sum-of-squares without unnecessary overflow.
/// Returns the values scl and ssq such that
/// \[
///     scl^2 ssq = x_1^2 + \dots + x_n^2 + scale^2 sumsq,
/// \]
/// where $x_i = | x( 1 + ( i - 1 )*incx ) |, 1 \le i \le n.$
/// The value of sumsq is
/// assumed to be at least unity and the value of ssq will then satisfy
///
///     1.0 <= ssq <= ( sumsq + 2*n ).
///
/// scale is assumed to be non-negative and scl returns the value
/// \[
///     scl = \max( scale, |real( x_i )|, |imag( x_i )| ).
/// \]
/// scale and sumsq must be supplied in scale and sumsq respectively.
/// scale and sumsq are overwritten by scl and ssq respectively.
///
/// The routine makes only one pass through the vector x.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] n
///     The number of elements to be used from the vector x.
///
/// @param[in] x
///     The vector x of length 1+(n-1)*incx.
///
/// @param[in] incx
///     The increment between successive values of the vector x.
///     incx > 0.
///
/// @param[in,out] scale
///     On entry, the value scale in the equation above.
///     On exit, scale is overwritten with the value scl.
///
/// @param[in,out] sumsq
///     On entry, the value sumsq in the equation above.
///     On exit, sumsq is overwritten with the value ssq.
///
/// @ingroup auxiliary
void lassq(
    int64_t n,
    std::complex<double> const* x, int64_t incx,
    double* scale,
    double* sumsq )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(incx) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int incx_ = (lapack_int) incx;

    LAPACK_zlassq(
        &n_,
        (lapack_complex_double*) x, &incx_, scale, sumsq );
}

}  // namespace lapack
