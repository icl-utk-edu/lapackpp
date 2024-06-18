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
void laev2(
    float a, float b, float c,
    float* rt1,
    float* rt2,
    float* cs1,
    float* sn1 )
{
    LAPACK_slaev2(
        &a, &b, &c, rt1, rt2, cs1, sn1 );
}

//------------------------------------------------------------------------------
/// @ingroup heev_computational
void laev2(
    double a, double b, double c,
    double* rt1,
    double* rt2,
    double* cs1,
    double* sn1 )
{
    LAPACK_dlaev2(
        &a, &b, &c, rt1, rt2, cs1, sn1 );
}

//------------------------------------------------------------------------------
/// @ingroup heev_computational
void laev2(
    std::complex<float> a, std::complex<float> b, std::complex<float> c,
    float* rt1,
    float* rt2,
    float* cs1,
    std::complex<float>* sn1 )
{
    LAPACK_claev2(
        (lapack_complex_float*) &a,
        (lapack_complex_float*) &b,
        (lapack_complex_float*) &c,
        rt1, rt2, cs1,
        (lapack_complex_float*) sn1 );
}

//------------------------------------------------------------------------------
/// Computes the eigendecomposition of a 2-by-2 Hermitian matrix
///     [ a          b ]
///     [ conj( b )  c ].
///
/// On return, rt1 is the eigenvalue of larger absolute value, rt2 is the
/// eigenvalue of smaller absolute value, and (cs1, sn1) is the unit right
/// eigenvector for rt1, giving the decomposition
///     [  cs1  conj( sn1 ) ] [ a          b ] [ cs1  -conj( sn1 ) ] = [ rt1  0   ]
///     [ -sn1  cs1         ] [ conj( b )  c ] [ sn1  cs1          ]   [ 0    rt2 ].
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] a
///     The (1, 1) element of the 2-by-2 matrix.
///
/// @param[in] b
///     The (1, 2) element and the conjugate of the (2, 1) element of
///     the 2-by-2 matrix.
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
/// @param[out] cs1
///
/// @param[out] sn1
///     The vector (cs1, sn1) is a unit right eigenvector for rt1.
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
/// cs1 and sn1 are accurate to a few ulps barring over/underflow.
///
/// Overflow is possible only if rt1 is within a factor of 5 of overflow.
/// Underflow is harmless if the input data is 0 or exceeds
/// underflow_threshold / macheps.
///
/// @ingroup heev_computational
void laev2(
    std::complex<double> a, std::complex<double> b, std::complex<double> c,
    double* rt1,
    double* rt2,
    double* cs1,
    std::complex<double>* sn1 )
{
    LAPACK_zlaev2(
        (lapack_complex_double*) &a,
        (lapack_complex_double*) &b,
        (lapack_complex_double*) &c,
        rt1, rt2, cs1,
        (lapack_complex_double*) sn1 );
}

}  // namespace lapack
