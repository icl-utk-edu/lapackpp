// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup pbsv_computational
int64_t pbcon(
    lapack::Uplo uplo, int64_t n, int64_t kd,
    float const* AB, int64_t ldab, float anorm,
    float* rcond )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (3*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_spbcon(
        &uplo_, &n_, &kd_,
        AB, &ldab_, &anorm, rcond,
        &work[0],
        &iwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup pbsv_computational
int64_t pbcon(
    lapack::Uplo uplo, int64_t n, int64_t kd,
    double const* AB, int64_t ldab, double anorm,
    double* rcond )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (3*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_dpbcon(
        &uplo_, &n_, &kd_,
        AB, &ldab_, &anorm, rcond,
        &work[0],
        &iwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup pbsv_computational
int64_t pbcon(
    lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<float> const* AB, int64_t ldab, float anorm,
    float* rcond )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (2*n) );
    lapack::vector< float > rwork( (n) );

    LAPACK_cpbcon(
        &uplo_, &n_, &kd_,
        (lapack_complex_float*) AB, &ldab_, &anorm, rcond,
        (lapack_complex_float*) &work[0],
        &rwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Estimates the reciprocal of the condition number (in the
/// 1-norm) of a Hermitian positive definite band matrix using
/// the Cholesky factorization $A = U^H U$ or $A = L L^H$ computed by
/// `lapack::pbtrf`.
///
/// An estimate is obtained for $|| A^{-1} ||,$ and the reciprocal of the
/// condition number is computed as $\text{rcond} = 1 / (||A||_1 \cdot || A^{-1} ||_1).$
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangular factor stored in AB;
///     - lapack::Uplo::Lower: Lower triangular factor stored in AB.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] kd
///     - If uplo = Upper, the number of superdiagonals of the matrix A;
///     - if uplo = Lower, the number of subdiagonals.
///     - kd >= 0.
///
/// @param[in] AB
///     The n-by-n band matrix AB, stored in an ldab-by-n array.
///     The triangular factor U or L from the Cholesky factorization
///     $A = U^H U$ or $A = L L^H$ of the band matrix A, stored in the
///     first kd+1 rows of the array. The j-th column of U or L is
///     stored in the j-th column of the array AB as follows:
///     - if uplo = Upper, AB(kd+1+i-j,j) = U(i,j) for max(1,j-kd) <= i <= j;
///     - if uplo = Lower, AB(1+i-j,j) = L(i,j) for j <= i <= min(n,j+kd).
///
/// @param[in] ldab
///     The leading dimension of the array AB. ldab >= kd+1.
///
/// @param[in] anorm
///     The 1-norm (or infinity-norm) of the Hermitian band matrix A.
///
/// @param[out] rcond
///     The reciprocal of the condition number of the matrix A,
///     computed as rcond = 1/(anorm * ainv_norm), where ainv_norm is an
///     estimate of the 1-norm of inv(A) computed in this routine.
///
/// @return = 0: successful exit
///
/// @ingroup pbsv_computational
int64_t pbcon(
    lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<double> const* AB, int64_t ldab, double anorm,
    double* rcond )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (2*n) );
    lapack::vector< double > rwork( (n) );

    LAPACK_zpbcon(
        &uplo_, &n_, &kd_,
        (lapack_complex_double*) AB, &ldab_, &anorm, rcond,
        (lapack_complex_double*) &work[0],
        &rwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
