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
/// @ingroup ptsv_computational
int64_t pttrs(
    int64_t n, int64_t nrhs,
    float const* D,
    float const* E,
    float* B, int64_t ldb )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_spttrs(
        &n_, &nrhs_,
        D,
        E,
        B, &ldb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ptsv_computational
int64_t pttrs(
    int64_t n, int64_t nrhs,
    double const* D,
    double const* E,
    double* B, int64_t ldb )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_dpttrs(
        &n_, &nrhs_,
        D,
        E,
        B, &ldb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ptsv_computational
int64_t pttrs(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    float const* D,
    std::complex<float> const* E,
    std::complex<float>* B, int64_t ldb )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_cpttrs(
        &uplo_, &n_, &nrhs_,
        D,
        (lapack_complex_float*) E,
        (lapack_complex_float*) B, &ldb_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Solves a tridiagonal system of the form
/// \[
///     A X = B
/// \]
/// using the factorization $A = U^H D U$ or $A = L D L^H$ computed by `lapack::pttrf`.
/// D is a diagonal matrix specified in the vector D, U (or L) is a unit
/// bidiagonal matrix whose superdiagonal (subdiagonal) is specified in
/// the vector E, and X and B are n by nrhs matrices.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] uplo
///     Specifies the form of the factorization and whether the
///     vector E is the superdiagonal of the upper bidiagonal factor
///     U or the subdiagonal of the lower bidiagonal factor L.
///     - lapack::Uplo::Upper: $A = U^H D U,$ E is the superdiagonal of U
///     - lapack::Uplo::Lower: $A = L D L^H,$ E is the subdiagonal of L
///
/// @param[in] n
///     The order of the tridiagonal matrix A. n >= 0.
///
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrix B. nrhs >= 0.
///
/// @param[in] D
///     The vector D of length n.
///     The n diagonal elements of the diagonal matrix D from the
///     factorization $A = U^H D U$ or $A = L D L^H.$
///
/// @param[in] E
///     The vector E of length n-1.
///     - If uplo = Upper, the (n-1) superdiagonal elements of the unit
///     bidiagonal factor U from the factorization $A = U^H D U.$
///
///     - If uplo = Lower, the (n-1) subdiagonal elements of the unit
///     bidiagonal factor L from the factorization $A = L D L^H.$
///
/// @param[in,out] B
///     The n-by-nrhs matrix B, stored in an ldb-by-nrhs array.
///     On entry, the right hand side vectors B for the system of
///     linear equations.
///     On exit, the solution vectors, X.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,n).
///
/// @return = 0: successful exit
///
/// @ingroup ptsv_computational
int64_t pttrs(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    double const* D,
    std::complex<double> const* E,
    std::complex<double>* B, int64_t ldb )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_zpttrs(
        &uplo_, &n_, &nrhs_,
        D,
        (lapack_complex_double*) E,
        (lapack_complex_double*) B, &ldb_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
