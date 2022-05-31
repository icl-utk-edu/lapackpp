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
float lapy2(
    float x, float y )
{
    return LAPACK_slapy2( &x, &y );
}

// -----------------------------------------------------------------------------
/// Returns $\sqrt{ x^2 + y^2 },$ taking care not to cause unnecessary
/// overflow.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] x
///
/// @param[in] y
///     x and y specify the values x and y.
///
/// @ingroup auxiliary
double lapy2(
    double x, double y )
{
    return LAPACK_dlapy2( &x, &y );
}

}  // namespace lapack
