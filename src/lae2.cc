// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

//------------------------------------------------------------------------------
/// @ingroup heev_computational
void lae2(
    float a, float b, float c,
    float* rt1,
    float* rt2 )
{
    LAPACK_slae2(
        &a, &b, &c, rt1, rt2 );
}

//------------------------------------------------------------------------------
/// Computes the eigenvalues of a 2-by-2 symmetric matrix
///     [ a b ]
///     [ b c ].
/// On return, rt1 is the eigenvalue of larger absolute value, and rt2
/// is the eigenvalue of smaller absolute value.
///
/// Overloaded versions are available for
/// `float`, `double`.
///
/// @param[in] a
///     The (1, 1) element of the 2-by-2 matrix.
///
/// @param[in] b
///     The (1, 2) and (2, 1) elements of the 2-by-2 matrix.
///
/// @param[in] c
///     The (2, 2) element of the 2-by-2 matrix.
///
/// @param[out] rt1
///     The eigenvalue of larger absolute value.
///
/// @param[out] rt2
///     The eigenvalue of smaller absolute value.
///
//------------------------------------------------------------------------------
/// @par Further Details
///
/// rt1 is accurate to a few ulps barring over/underflow.
///
/// rt2 may be inaccurate if there is massive cancellation in the
/// determinant a*c-b*b; higher precision or correctly rounded or
/// correctly truncated arithmetic would be needed to compute rt2
/// accurately in all cases.
///
/// Overflow is possible only if rt1 is within a factor of 5 of overflow.
/// Underflow is harmless if the input data is 0 or exceeds
/// underflow_threshold / macheps.
///
/// @ingroup heev_computational
void lae2(
    double a, double b, double c,
    double* rt1,
    double* rt2 )
{
    LAPACK_dlae2(
        &a, &b, &c, rt1, rt2 );
}

}  // namespace lapack
