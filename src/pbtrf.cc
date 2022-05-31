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
int64_t pbtrf(
    lapack::Uplo uplo, int64_t n, int64_t kd,
    float* AB, int64_t ldab )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int info_ = 0;

    LAPACK_spbtrf(
        &uplo_, &n_, &kd_,
        AB, &ldab_, &info_
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
int64_t pbtrf(
    lapack::Uplo uplo, int64_t n, int64_t kd,
    double* AB, int64_t ldab )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int info_ = 0;

    LAPACK_dpbtrf(
        &uplo_, &n_, &kd_,
        AB, &ldab_, &info_
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
int64_t pbtrf(
    lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<float>* AB, int64_t ldab )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int info_ = 0;

    LAPACK_cpbtrf(
        &uplo_, &n_, &kd_,
        (lapack_complex_float*) AB, &ldab_, &info_
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
/// Computes the Cholesky factorization of a Hermitian
/// positive definite band matrix A.
///
/// The factorization has the form
///     $A = U^H U,$ if uplo = Upper, or
///     $A = L L^H,$ if uplo = Lower,
/// where U is an upper triangular matrix and L is lower triangular.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangle of A is stored;
///     - lapack::Uplo::Lower: Lower triangle of A is stored.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] kd
///     - If uplo = Upper, the number of superdiagonals of the matrix A;
///     - if uplo = Lower, the number of subdiagonals.
///     - kd >= 0.
///
/// @param[in,out] AB
///     The n-by-n band matrix AB, stored in an ldab-by-n array.
///     - On entry, the upper or lower triangle of the Hermitian band
///     matrix A, stored in the first kd+1 rows of the array. The
///     j-th column of A is stored in the j-th column of the array AB
///     as follows:
///       - if uplo = Upper, AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd) <= i <= j;
///       - if uplo = Lower, AB(1+i-j,j) = A(i,j) for j <= i <= min(n,j+kd).
///
///     - On successful exit, the triangular factor U or L from the
///     Cholesky factorization $A = U^H U$ or $A = L L^H$ of the band
///     matrix A, in the same storage format as A.
///
/// @param[in] ldab
///     The leading dimension of the array AB. ldab >= kd+1.
///
/// @return = 0: successful exit
/// @return > 0: if return value = i, the leading minor of order i is not
///     positive definite, and the factorization could not be
///     completed.
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
/// @ingroup pbsv_computational
int64_t pbtrf(
    lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<double>* AB, int64_t ldab )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int info_ = 0;

    LAPACK_zpbtrf(
        &uplo_, &n_, &kd_,
        (lapack_complex_double*) AB, &ldab_, &info_
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
