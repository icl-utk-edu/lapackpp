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
/// @ingroup pbsv_computational
int64_t pbtrs(
    lapack::Uplo uplo, int64_t n, int64_t kd, int64_t nrhs,
    float const* AB, int64_t ldab,
    float* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int nrhs_ = (lapack_int) nrhs;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;

    LAPACK_spbtrs(
        &uplo_, &n_, &kd_, &nrhs_,
        AB, &ldab_,
        B, &ldb_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup pbsv_computational
int64_t pbtrs(
    lapack::Uplo uplo, int64_t n, int64_t kd, int64_t nrhs,
    double const* AB, int64_t ldab,
    double* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int nrhs_ = (lapack_int) nrhs;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;

    LAPACK_dpbtrs(
        &uplo_, &n_, &kd_, &nrhs_,
        AB, &ldab_,
        B, &ldb_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup pbsv_computational
int64_t pbtrs(
    lapack::Uplo uplo, int64_t n, int64_t kd, int64_t nrhs,
    std::complex<float> const* AB, int64_t ldab,
    std::complex<float>* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int nrhs_ = (lapack_int) nrhs;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;

    LAPACK_cpbtrs(
        &uplo_, &n_, &kd_, &nrhs_,
        (lapack_complex_float*) AB, &ldab_,
        (lapack_complex_float*) B, &ldb_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Solves a system of linear equations $A X = B$ with a Hermitian
/// positive definite band matrix A using the Cholesky factorization
/// $A = U^H U$ or $A = L L^H$ computed by `lapack::pbtrf`.
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
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrix B. nrhs >= 0.
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
/// @param[in,out] B
///     The n-by-nrhs matrix B, stored in an ldb-by-nrhs array.
///     On entry, the right hand side matrix B.
///     On exit, the solution matrix X.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,n).
///
/// @return = 0: successful exit
///
/// @ingroup pbsv_computational
int64_t pbtrs(
    lapack::Uplo uplo, int64_t n, int64_t kd, int64_t nrhs,
    std::complex<double> const* AB, int64_t ldab,
    std::complex<double>* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int nrhs_ = (lapack_int) nrhs;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;

    LAPACK_zpbtrs(
        &uplo_, &n_, &kd_, &nrhs_,
        (lapack_complex_double*) AB, &ldab_,
        (lapack_complex_double*) B, &ldb_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
