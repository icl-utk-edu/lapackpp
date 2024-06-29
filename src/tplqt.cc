// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#if LAPACK_VERSION >= 30700  // >= 3.7.0

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup tplqt
int64_t tplqt(
    int64_t m, int64_t n, int64_t l, int64_t mb,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    float* T, int64_t ldt )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int l_ = to_lapack_int( l );
    lapack_int mb_ = to_lapack_int( mb );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (mb*m) );

    LAPACK_stplqt(
        &m_, &n_, &l_, &mb_,
        A, &lda_,
        B, &ldb_,
        T, &ldt_,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup tplqt
int64_t tplqt(
    int64_t m, int64_t n, int64_t l, int64_t mb,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    double* T, int64_t ldt )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int l_ = to_lapack_int( l );
    lapack_int mb_ = to_lapack_int( mb );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (mb*m) );

    LAPACK_dtplqt(
        &m_, &n_, &l_, &mb_,
        A, &lda_,
        B, &ldb_,
        T, &ldt_,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup tplqt
int64_t tplqt(
    int64_t m, int64_t n, int64_t l, int64_t mb,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* T, int64_t ldt )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int l_ = to_lapack_int( l );
    lapack_int mb_ = to_lapack_int( mb );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (mb*m) );

    LAPACK_ctplqt(
        &m_, &n_, &l_, &mb_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) T, &ldt_,
        (lapack_complex_float*) &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes a blocked LQ factorization of a complex
/// "triangular-pentagonal" matrix C, which is composed of a
/// triangular block A and pentagonal block B, using the compact
/// WY representation for Q.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @since LAPACK 3.7.0
///
/// @param[in] m
///     The number of rows of the matrix B, and the order of the
///     triangular matrix A.
///     m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix B.
///     n >= 0.
///
/// @param[in] l
///     The number of rows of the lower trapezoidal part of B.
///     min(m,n) >= l >= 0. See Further Details.
///
/// @param[in] mb
///     The block size to be used in the blocked QR. m >= mb >= 1.
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
///     The mb-by-n matrix T, stored in an ldt-by-n array.
///     The lower triangular block reflectors stored in compact form
///     as a sequence of upper triangular blocks. See Further Details.
///
/// @param[in] ldt
///     The leading dimension of the array T. ldt >= mb.
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
/// m-by-m lower triangular matrix, where 0 <= l <= min(m,n). If l=0,
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
///
/// The number of blocks is B = ceiling(m/mb), where each
/// block is of order mb except for the last block, which is of order
/// IB = m - (m-1)*mb. For each of the B blocks, a upper triangular block
/// reflector factor is computed: T1, T2, ..., TB. The mb-by-mb (and IB-by-IB
/// for the last block) T's are stored in the mb-by-n matrix T as
/// \[
///     T = [T1, T2, ..., TB].
/// \]
///
/// @ingroup tplqt
int64_t tplqt(
    int64_t m, int64_t n, int64_t l, int64_t mb,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* T, int64_t ldt )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int l_ = to_lapack_int( l );
    lapack_int mb_ = to_lapack_int( mb );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (mb*m) );

    LAPACK_ztplqt(
        &m_, &n_, &l_, &mb_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) T, &ldt_,
        (lapack_complex_double*) &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.7.0
