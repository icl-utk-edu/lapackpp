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
/// @ingroup ptsv_computational
int64_t ptrfs(
    int64_t n, int64_t nrhs,
    float const* D,
    float const* E,
    float const* DF,
    float const* EF,
    float const* B, int64_t ldb,
    float* X, int64_t ldx,
    float* ferr,
    float* berr )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (2*n) );

    LAPACK_sptrfs(
        &n_, &nrhs_,
        D,
        E,
        DF,
        EF,
        B, &ldb_,
        X, &ldx_,
        ferr,
        berr,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ptsv_computational
int64_t ptrfs(
    int64_t n, int64_t nrhs,
    double const* D,
    double const* E,
    double const* DF,
    double const* EF,
    double const* B, int64_t ldb,
    double* X, int64_t ldx,
    double* ferr,
    double* berr )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (2*n) );

    LAPACK_dptrfs(
        &n_, &nrhs_,
        D,
        E,
        DF,
        EF,
        B, &ldb_,
        X, &ldx_,
        ferr,
        berr,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ptsv_computational
int64_t ptrfs(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    float const* D,
    std::complex<float> const* E,
    float const* DF,
    std::complex<float> const* EF,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float>* X, int64_t ldx,
    float* ferr,
    float* berr )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (n) );
    lapack::vector< float > rwork( (n) );

    LAPACK_cptrfs(
        &uplo_, &n_, &nrhs_,
        D,
        (lapack_complex_float*) E,
        DF,
        (lapack_complex_float*) EF,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) X, &ldx_,
        ferr,
        berr,
        (lapack_complex_float*) &work[0],
        &rwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Improves the computed solution to a system of linear
/// equations when the coefficient matrix is Hermitian positive definite
/// and tridiagonal, and provides error bounds and backward error
/// estimates for the solution.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] uplo
///     Whether the superdiagonal or the subdiagonal of the
///     tridiagonal matrix A is stored and the form of the
///     factorization:
///     - lapack::Uplo::Upper: E is the superdiagonal of A, and $A = U^H D U;$
///     - lapack::Uplo::Lower: E is the subdiagonal   of A, and $A = L D L^H.$
///     \n
///     (The two forms are equivalent if A is real.)
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrix B. nrhs >= 0.
///
/// @param[in] D
///     The vector D of length n.
///     The n real diagonal elements of the tridiagonal matrix A.
///
/// @param[in] E
///     The vector E of length n-1.
///     The (n-1) off-diagonal elements of the tridiagonal matrix A
///     (see uplo).
///
/// @param[in] DF
///     The vector DF of length n.
///     The n diagonal elements of the diagonal matrix D from
///     the factorization computed by `lapack::pttrf`.
///
/// @param[in] EF
///     The vector EF of length n-1.
///     The (n-1) off-diagonal elements of the unit bidiagonal
///     factor U or L from the factorization computed by `lapack::pttrf`
///     (see uplo).
///
/// @param[in] B
///     The n-by-nrhs matrix B, stored in an ldb-by-nrhs array.
///     The right hand side matrix B.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,n).
///
/// @param[in,out] X
///     The n-by-nrhs matrix X, stored in an ldx-by-nrhs array.
///     On entry, the solution matrix X, as computed by `lapack::pttrs`.
///     On exit, the improved solution matrix X.
///
/// @param[in] ldx
///     The leading dimension of the array X. ldx >= max(1,n).
///
/// @param[out] ferr
///     The vector ferr of length nrhs.
///     The forward error bound for each solution vector
///     X(j) (the j-th column of the solution matrix X).
///     If XTRUE is the true solution corresponding to X(j), ferr(j)
///     is an estimated upper bound for the magnitude of the largest
///     element in (X(j) - XTRUE) divided by the magnitude of the
///     largest element in X(j).
///
/// @param[out] berr
///     The vector berr of length nrhs.
///     The componentwise relative backward error of each solution
///     vector X(j) (i.e., the smallest relative change in
///     any element of A or B that makes X(j) an exact solution).
///
/// @return = 0: successful exit
///
/// @ingroup ptsv_computational
int64_t ptrfs(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    double const* D,
    std::complex<double> const* E,
    double const* DF,
    std::complex<double> const* EF,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double>* X, int64_t ldx,
    double* ferr,
    double* berr )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (n) );
    lapack::vector< double > rwork( (n) );

    LAPACK_zptrfs(
        &uplo_, &n_, &nrhs_,
        D,
        (lapack_complex_double*) E,
        DF,
        (lapack_complex_double*) EF,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) X, &ldx_,
        ferr,
        berr,
        (lapack_complex_double*) &work[0],
        &rwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
