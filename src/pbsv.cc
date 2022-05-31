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
/// @ingroup pbsv
int64_t pbsv(
    lapack::Uplo uplo, int64_t n, int64_t kd, int64_t nrhs,
    float* AB, int64_t ldab,
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

    LAPACK_spbsv(
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
/// @ingroup pbsv
int64_t pbsv(
    lapack::Uplo uplo, int64_t n, int64_t kd, int64_t nrhs,
    double* AB, int64_t ldab,
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

    LAPACK_dpbsv(
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
/// @ingroup pbsv
int64_t pbsv(
    lapack::Uplo uplo, int64_t n, int64_t kd, int64_t nrhs,
    std::complex<float>* AB, int64_t ldab,
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

    LAPACK_cpbsv(
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
/// Computes the solution to a system of linear equations
/// \[
///     A X = B,
/// \]
/// where A is an n-by-n Hermitian positive definite band matrix and X
/// and B are n-by-nrhs matrices.
///
/// The Cholesky decomposition is used to factor A as
///     $A = U^H U,$ if uplo = Upper, or
///     $A = L L^H,$ if uplo = Lower,
/// where U is an upper triangular band matrix, and L is a lower
/// triangular band matrix, with the same number of superdiagonals or
/// subdiagonals as A. The factored form of A is then used to solve the
/// system of equations $A X = B.$
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangle of A is stored;
///     - lapack::Uplo::Lower: Lower triangle of A is stored.
///
/// @param[in] n
///     The number of linear equations, i.e., the order of the
///     matrix A. n >= 0.
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
/// @param[in,out] AB
///     The n-by-n band matrix AB, stored in an ldab-by-n array.
///     - On entry, the upper or lower triangle of the Hermitian band
///     matrix A, stored in the first kd+1 rows of the array. The
///     j-th column of A is stored in the j-th column of the array AB
///     as follows:
///       - if uplo = Upper, AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd) <= i <= j;
///       - if uplo = Lower, AB(1+i-j,j) = A(i,j) for j <= i <= min(n,j+kd).
///       \n
///       See below for further details.
///
///     - On successful exit, the triangular factor U or L from the
///     Cholesky factorization $A = U^H U$ or $A = L L^H$ of the band
///     matrix A, in the same storage format as A.
///
/// @param[in] ldab
///     The leading dimension of the array AB. ldab >= kd+1.
///
/// @param[in,out] B
///     The n-by-nrhs matrix B, stored in an ldb-by-nrhs array.
///     On entry, the n-by-nrhs right hand side matrix B.
///     On successful exit, the n-by-nrhs solution matrix X.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,n).
///
/// @return = 0: successful exit
/// @return > 0: if return value = i, the leading minor of order i of A is not
///     positive definite, so the factorization could not be
///     completed, and the solution has not been computed.
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The band storage scheme is illustrated by the following example, when
/// n = 6, kd = 2, and uplo = Upper:
///
///     On entry:                        On exit:
///
///      *    *   a13  a24  a35  a46      *    *   u13  u24  u35  u46
///      *   a12  a23  a34  a45  a56      *   u12  u23  u34  u45  u56
///     a11  a22  a33  a44  a55  a66     u11  u22  u33  u44  u55  u66
///
/// Similarly, if uplo = Lower the format of A is as follows:
///
///     On entry:                        On exit:
///
///     a11  a22  a33  a44  a55  a66     l11  l22  l33  l44  l55  l66
///     a21  a32  a43  a54  a65   *      l21  l32  l43  l54  l65   *
///     a31  a42  a53  a64   *    *      l31  l42  l53  l64   *    *
///
/// Array elements marked * are not used by the routine.
///
/// @ingroup pbsv
int64_t pbsv(
    lapack::Uplo uplo, int64_t n, int64_t kd, int64_t nrhs,
    std::complex<double>* AB, int64_t ldab,
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

    LAPACK_zpbsv(
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
