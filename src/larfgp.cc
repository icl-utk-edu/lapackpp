// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"

#if LAPACK_VERSION >= 30202  // >= 3.2.2

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup reflector_aux_grp
void larfgp(
    int64_t n,
    float* alpha,
    float* X, int64_t incx,
    float* tau )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int incx_ = to_lapack_int( incx );

    LAPACK_slarfgp(
        &n_, alpha,
        X, &incx_, tau );
}

// -----------------------------------------------------------------------------
/// @ingroup reflector_aux_grp
void larfgp(
    int64_t n,
    double* alpha,
    double* X, int64_t incx,
    double* tau )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int incx_ = to_lapack_int( incx );

    LAPACK_dlarfgp(
        &n_, alpha,
        X, &incx_, tau );
}

// -----------------------------------------------------------------------------
/// @ingroup reflector_aux_grp
void larfgp(
    int64_t n,
    std::complex<float>* alpha,
    std::complex<float>* X, int64_t incx,
    std::complex<float>* tau )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int incx_ = to_lapack_int( incx );

    LAPACK_clarfgp(
        &n_, (lapack_complex_float*) alpha,
        (lapack_complex_float*) X, &incx_, (lapack_complex_float*) tau );
}

// -----------------------------------------------------------------------------
/// Generates an elementary reflector H of order n, such that:
/// \[
///     H^H
///     \begin{bmatrix}
///             \alpha
///         \\  x
///     \end{bmatrix}
///     =
///     \begin{bmatrix}
///             \beta
///         \\  0
///     \end{bmatrix};
///     \quad
///     H^H H = I.
/// \]
/// where $\alpha$ and $\beta$ are scalars, with $\beta$ real and non-negative,
/// and x is an (n-1)-element vector. H is represented in the form
/// \[
///     H = I - \tau
///     \begin{bmatrix}
///             1
///         \\  v
///     \end{bmatrix}
///     \begin{bmatrix}
///         1  &  v^H
///     \end{bmatrix},
/// \]
/// where $\tau$ is a scalar and v is a (n-1)-element
/// vector. For complex H, note that H is not hermitian.
///
/// If the elements of x are all zero and alpha is real, then $\tau = 0$
/// and H is taken to be the unit matrix.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @since LAPACK 3.2.2
///
/// @param[in] n
///     The order of the elementary reflector.
///
/// @param[in,out] alpha
///     On entry, the value alpha.
///     On exit, it is overwritten with the value beta.
///
/// @param[in,out] X
///     The vector X of length 1+(n-2)*abs(incx).
///     On entry, the vector x.
///     On exit, it is overwritten with the vector v.
///
/// @param[in] incx
///     The increment between elements of X. incx > 0.
///
/// @param[out] tau
///     The value tau.
///
/// @ingroup reflector_aux_grp
void larfgp(
    int64_t n,
    std::complex<double>* alpha,
    std::complex<double>* X, int64_t incx,
    std::complex<double>* tau )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int incx_ = to_lapack_int( incx );

    LAPACK_zlarfgp(
        &n_, (lapack_complex_double*) alpha,
        (lapack_complex_double*) X, &incx_, (lapack_complex_double*) tau );
}

}  // namespace lapack

#endif  // LAPACK >= 3.2.2
