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
/// @ingroup heev_auxiliary
int64_t laed4(
    int64_t n, int64_t i,
    float const* d,
    float const* z,
    float* delta, float rho,
    float* lambda )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int i_ = to_lapack_int( i ) + 1;  // change to 1-based
    lapack_int info_ = 0;

    LAPACK_slaed4(
        &n_, &i_,
        d,
        z,
        delta, &rho, lambda, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
///
/// This subroutine computes the i-th updated eigenvalue of a symmetric
/// rank-one modification to a diagonal matrix whose elements are
/// given in the array d, and that
/// \[
///     d(i) < d(j) for i < j
/// \]
/// and that rho > 0. This is arranged by the calling routine, and is
/// no loss in generality. The rank-one modified system is thus
/// \[
///     diag( d ) + \rho z z^T
/// \]
/// where we assume the Euclidean norm of z is 1.
///
/// The method consists of approximating the rational functions in the
/// secular equation by simpler interpolating rational functions.
///
/// Overloaded versions are available for
/// `float`, `double`.
///
/// @param[in] n
///     The length of all arrays.
///
/// @param[in] i
///     The index of the eigenvalue to be computed. 0 <= i < n.
///     Unlike LAPACK, here this is 0-based.
///
/// @param[in] d
///     The vector d of length n.
///     The original eigenvalues. It is assumed that they are in
///     order, d(i) < d(j) for i < j.
///
/// @param[in] z
///     The vector z of length n.
///     The components of the updating vector.
///
/// @param[out] delta
///     The vector delta of length n.
///     If n > 2, delta contains (d(j) - lambda_i) in its j-th
///     component. If n = 1, then delta(1) = 1. If n = 2, see `lapack::laed5`
///     for detail. The vector delta contains the information necessary
///     to construct the eigenvectors by `lapack::laed3` and `lapack::laed9`.
///
/// @param[in] rho
///     The scalar in the symmetric updating formula.
///
/// @param[out] lambda
///     The computed lambda_i, the i-th updated eigenvalue.
///
/// @retval = 0: successful exit
/// @retval > 0: if return value = 1, the updating process failed.
///
/// @ingroup heev_auxiliary
int64_t laed4(
    int64_t n, int64_t i,
    double const* d,
    double const* z,
    double* delta, double rho,
    double* lambda )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int i_ = to_lapack_int( i ) + 1;  // change to 1-based
    lapack_int info_ = 0;

    LAPACK_dlaed4(
        &n_, &i_,
        d,
        z,
        delta, &rho, lambda, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
