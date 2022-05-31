// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup ptsv_computational
int64_t ptcon(
    int64_t n,
    float const* D,
    float const* E, float anorm,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (n) );

    LAPACK_sptcon(
        &n_,
        D,
        E, &anorm, rcond,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ptsv_computational
int64_t ptcon(
    int64_t n,
    double const* D,
    double const* E, double anorm,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (n) );

    LAPACK_dptcon(
        &n_,
        D,
        E, &anorm, rcond,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ptsv_computational
int64_t ptcon(
    int64_t n,
    float const* D,
    std::complex<float> const* E, float anorm,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > rwork( (n) );

    LAPACK_cptcon(
        &n_,
        D,
        (lapack_complex_float*) E, &anorm, rcond,
        &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the reciprocal of the condition number (in the
/// 1-norm) of a Hermitian positive definite tridiagonal matrix
/// using the factorization $A = L D L^H$ or $A = U^H D U$ computed by
/// `lapack::pttrf`.
///
/// $|| A^{-1} ||$ is computed by a direct method, and the reciprocal of
/// the condition number is computed as
/// \[
///     \text{rcond} = 1 / (||A||_1 \cdot ||A^{-1}||_1).
/// \]
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] D
///     The vector D of length n.
///     The n diagonal elements of the diagonal matrix D from the
///     factorization of A, as computed by `lapack::pttrf`.
///
/// @param[in] E
///     The vector E of length n-1.
///     The (n-1) off-diagonal elements of the unit bidiagonal factor
///     U or L from the factorization of A, as computed by `lapack::pttrf`.
///
/// @param[in] anorm
///     The 1-norm of the original matrix A.
///
/// @param[out] rcond
///     The reciprocal of the condition number of the matrix A,
///     computed as rcond = 1/(anorm * ainv_norm), where ainv_norm is the
///     1-norm of $A^{-1}$ computed in this routine.
///
/// @return = 0: successful exit
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The method used is described in Nicholas J. Higham, "Efficient
/// Algorithms for Computing the Condition Number of a Tridiagonal
/// Matrix", SIAM J. Sci. Stat. Comput., Vol. 7, No. 1, January 1986.
///
/// @ingroup ptsv_computational
int64_t ptcon(
    int64_t n,
    double const* D,
    std::complex<double> const* E, double anorm,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > rwork( (n) );

    LAPACK_zptcon(
        &n_,
        D,
        (lapack_complex_double*) E, &anorm, rcond,
        &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
