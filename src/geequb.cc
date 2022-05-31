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
/// @ingroup gesv_computational
int64_t geequb(
    int64_t m, int64_t n,
    float const* A, int64_t lda,
    float* R,
    float* C,
    float* rowcnd,
    float* colcnd,
    float* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    LAPACK_sgeequb(
        &m_, &n_,
        A, &lda_,
        R,
        C, rowcnd, colcnd, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesv_computational
int64_t geequb(
    int64_t m, int64_t n,
    double const* A, int64_t lda,
    double* R,
    double* C,
    double* rowcnd,
    double* colcnd,
    double* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    LAPACK_dgeequb(
        &m_, &n_,
        A, &lda_,
        R,
        C, rowcnd, colcnd, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesv_computational
int64_t geequb(
    int64_t m, int64_t n,
    std::complex<float> const* A, int64_t lda,
    float* R,
    float* C,
    float* rowcnd,
    float* colcnd,
    float* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    LAPACK_cgeequb(
        &m_, &n_,
        (lapack_complex_float*) A, &lda_,
        R,
        C, rowcnd, colcnd, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes row and column scalings intended to equilibrate an
/// m-by-n matrix A and reduce its condition number. R returns the row
/// scale factors and C the column scale factors, chosen to try to make
/// the largest element in each row and column of the matrix B with
/// elements $B_{i,j} = R_{i} A_{i,j} C_{j}$ have an absolute value of at most
/// the radix.
///
/// $R_{i}$ and $C_{j}$ are restricted to be a power of the radix between
/// smlnum = smallest safe number and bignum = largest safe number. Use
/// of these scaling factors is not guaranteed to reduce the condition
/// number of A but works well in practice.
///
/// This routine differs from `lapack::geequ` by restricting the scaling factors
/// to a power of the radix. Barring over- and underflow, scaling by
/// these factors introduces no additional rounding errors. However, the
/// scaled entries' magnitudes are no longer approximately 1 but lie
/// between sqrt(radix) and 1/sqrt(radix).
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] m
///     The number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix A. n >= 0.
///
/// @param[in] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     The m-by-n matrix whose equilibration factors are
///     to be computed.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @param[out] R
///     The vector R of length m.
///     If successful or return value > m, R contains the row scale factors
///     for A.
///
/// @param[out] C
///     The vector C of length n.
///     If successful, C contains the column scale factors for A.
///
/// @param[out] rowcnd
///     If successful or return value > m, rowcnd contains the ratio of the
///     smallest R(i) to the largest R(i). If rowcnd >= 0.1 and
///     amax is neither too large nor too small, it is not worth
///     scaling by R.
///
/// @param[out] colcnd
///     If successful, colcnd contains the ratio of the smallest
///     C(i) to the largest C(i). If colcnd >= 0.1, it is not
///     worth scaling by C.
///
/// @param[out] amax
///     Absolute value of largest matrix element. If amax is very
///     close to overflow or very close to underflow, the matrix
///     should be scaled.
///
/// @return = 0: successful exit
/// @return > 0 and <= m: if return value = i, the i-th row of A is exactly zero
/// @return > m:          if return value = i, the (i-m)-th column of A is exactly zero
///
/// @ingroup gesv_computational
int64_t geequb(
    int64_t m, int64_t n,
    std::complex<double> const* A, int64_t lda,
    double* R,
    double* C,
    double* rowcnd,
    double* colcnd,
    double* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    LAPACK_zgeequb(
        &m_, &n_,
        (lapack_complex_double*) A, &lda_,
        R,
        C, rowcnd, colcnd, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
