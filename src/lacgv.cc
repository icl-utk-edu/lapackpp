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
void lacgv(
    int64_t n,
    std::complex<float>* x, int64_t incx )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(incx) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int incx_ = (lapack_int) incx;

    LAPACK_clacgv(
        &n_,
        (lapack_complex_float*) x, &incx_ );
}

// -----------------------------------------------------------------------------
/// Conjugates a complex vector of length n.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// Real precisions are dummy inline functions that do nothing,
/// to facilitate templating.
///
/// @param[in] n
///     The length of the vector x. n >= 0.
///
/// @param[in,out] x
///     The vector x of length n, stored in an array of length 1+(n-1)*abs(incx).
///     On entry, the vector of length n to be conjugated.
///     On exit, x is overwritten with conj(x).
///
/// @param[in] incx
///     The spacing between successive elements of x.
///
/// @ingroup auxiliary
void lacgv(
    int64_t n,
    std::complex<double>* x, int64_t incx )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(incx) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int incx_ = (lapack_int) incx;

    LAPACK_zlacgv(
        &n_,
        (lapack_complex_double*) x, &incx_ );
}

}  // namespace lapack
