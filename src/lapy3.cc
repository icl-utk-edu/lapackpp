// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup auxiliary
float lapy3(
    float x, float y, float z )
{
    return LAPACK_slapy3( &x, &y, &z );
}

// -----------------------------------------------------------------------------
/// Returns $\sqrt{ x^2 + y^2 + z^2 },$ taking care not to cause
/// unnecessary overflow.
///
/// Overloaded versions are available for
/// `float`, `double`.
///
/// @param[in] x
///
/// @param[in] y
///
/// @param[in] z
///     x, y and z specify the values x, y and z.
///
/// @ingroup auxiliary
double lapy3(
    double x, double y, double z )
{
    return LAPACK_dlapy3( &x, &y, &z );
}

}  // namespace lapack
