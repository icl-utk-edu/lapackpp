// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"

#if LAPACK_VERSION >= 30700  // >= 3.7.0

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup tplqt
int64_t tplqt2(
    int64_t m, int64_t n, int64_t l,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    float* T, int64_t ldt )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int l_ = to_lapack_int( l );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int info_ = 0;

    LAPACK_stplqt2(
        &m_, &n_, &l_,
        A, &lda_,
        B, &ldb_,
        T, &ldt_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup tplqt
int64_t tplqt2(
    int64_t m, int64_t n, int64_t l,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    double* T, int64_t ldt )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int l_ = to_lapack_int( l );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int info_ = 0;

    LAPACK_dtplqt2(
        &m_, &n_, &l_,
        A, &lda_,
        B, &ldb_,
        T, &ldt_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup tplqt
int64_t tplqt2(
    int64_t m, int64_t n, int64_t l,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* T, int64_t ldt )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int l_ = to_lapack_int( l );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int info_ = 0;

    LAPACK_ctplqt2(
        &m_, &n_, &l_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) T, &ldt_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes a LQ a factorization of a complex "triangular-pentagonal"
/// matrix C, which is composed of a triangular block A and pentagonal block B,
/// using the compact WY representation for Q.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @since LAPACK 3.7.0
///
/// @param[in] m
///     The total number of rows of the matrix B.
///     m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix B, and the order of
///     the triangular matrix A.
///     n >= 0.
///
/// @param[in] l
///     The number of rows of the lower trapezoidal part of B.
///     min(m,n) >= l >= 0. See Further Details.
///
/// @param[in,out] A
///     The m-by-m matrix A, stored in an lda-by-m array.
///     On entry, the lower triangular m-by-m matrix A.
///     On exit, the elements on and below the diagonal of the array
///     contain the lower triangular matrix l.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @param[in,out] B
///     The m-by-n matrix B, stored in an ldb-by-n array.
///     On entry, the pentagonal m-by-n matrix B. The first n-l columns
///     are rectangular, and the last l columns are lower trapezoidal.
///     On exit, B contains the pentagonal matrix V. See Further Details.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,m).
///
/// @param[out] T
///     The m-by-m matrix T, stored in an ldt-by-m array.
///     The n-by-n upper triangular factor T of the block reflector.
///     See Further Details.
///
/// @param[in] ldt
///     The leading dimension of the array T. ldt >= max(1,m)
///
/// @return = 0: successful exit
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The input matrix C is a m-by-(m+n) matrix
/// \[
///     C = [ A, B ]
/// \]
/// where A is an lower triangular m-by-m matrix, and B is m-by-n pentagonal
/// matrix consisting of a m-by-(n-l) rectangular matrix B1 left of a m-by-l
/// upper trapezoidal matrix B2:
/// \[
///     B = [ B1, B2 ]
/// \]
/// The lower trapezoidal matrix B2 consists of the first l columns of a
/// n-by-n lower triangular matrix, where 0 <= l <= min(m,n). If l=0,
/// B is rectangular m-by-n; if m=l=n, B is lower triangular.
///
/// The matrix W stores the elementary reflectors H(i) in the i-th row
/// above the diagonal (of A) in the m-by-(m+n) input matrix C
/// so that W can be represented as
/// \[
///     W = [ I, V ]
/// \]
/// where I is m-by-m identity and V is m-by-n, same form as B.
///
/// Thus, all of information needed for W is contained on exit in B, which
/// we call V above. Note that V has the same form as B; that is,
/// \[
///     V = [ V1, V2 ]
/// \]
/// where V1 is m-by-(n-l) rectangular, V2 is m-by-l lower trapezoidal.
/// The rows of V represent the vectors which define the H(i)'s.
/// The (m+n)-by-(m+n) block reflector H is then given by
/// \[
///     H = I - W^T T W
/// \]
/// where W^H is the conjugate transpose of W and T is the upper triangular
/// factor of the block reflector.
///
/// @ingroup tplqt
int64_t tplqt2(
    int64_t m, int64_t n, int64_t l,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* T, int64_t ldt )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int l_ = to_lapack_int( l );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int info_ = 0;

    LAPACK_ztplqt2(
        &m_, &n_, &l_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) T, &ldt_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.7.0
