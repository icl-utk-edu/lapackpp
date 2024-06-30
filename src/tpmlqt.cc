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
int64_t tpmlqt(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t k, int64_t l, int64_t mb,
    float const* V, int64_t ldv,
    float const* T, int64_t ldt,
    float* A, int64_t lda,
    float* B, int64_t ldb )
{
    // for real, map ConjTrans to Trans
    if (trans == Op::ConjTrans)
        trans = Op::Trans;

    char side_ = to_char( side );
    char trans_ = to_char( trans );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int k_ = to_lapack_int( k );
    lapack_int l_ = to_lapack_int( l );
    lapack_int mb_ = to_lapack_int( mb );
    lapack_int ldv_ = to_lapack_int( ldv );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    // allocate workspace
    int64_t lwork = (side == Side::Left ? n*mb : m*mb);
    lapack::vector< float > work( lwork );

    LAPACK_stpmlqt(
        &side_, &trans_, &m_, &n_, &k_, &l_, &mb_,
        V, &ldv_,
        T, &ldt_,
        A, &lda_,
        B, &ldb_,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup tplqt
int64_t tpmlqt(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t k, int64_t l, int64_t mb,
    double const* V, int64_t ldv,
    double const* T, int64_t ldt,
    double* A, int64_t lda,
    double* B, int64_t ldb )
{
    // for real, map ConjTrans to Trans
    if (trans == Op::ConjTrans)
        trans = Op::Trans;

    char side_ = to_char( side );
    char trans_ = to_char( trans );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int k_ = to_lapack_int( k );
    lapack_int l_ = to_lapack_int( l );
    lapack_int mb_ = to_lapack_int( mb );
    lapack_int ldv_ = to_lapack_int( ldv );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    // allocate workspace
    int64_t lwork = (side == Side::Left ? n*mb : m*mb);
    lapack::vector< double > work( lwork );

    LAPACK_dtpmlqt(
        &side_, &trans_, &m_, &n_, &k_, &l_, &mb_,
        V, &ldv_,
        T, &ldt_,
        A, &lda_,
        B, &ldb_,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup tplqt
int64_t tpmlqt(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t k, int64_t l, int64_t mb,
    std::complex<float> const* V, int64_t ldv,
    std::complex<float> const* T, int64_t ldt,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb )
{
    char side_ = to_char( side );
    char trans_ = to_char( trans );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int k_ = to_lapack_int( k );
    lapack_int l_ = to_lapack_int( l );
    lapack_int mb_ = to_lapack_int( mb );
    lapack_int ldv_ = to_lapack_int( ldv );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    // allocate workspace
    int64_t lwork = (side == Side::Left ? n*mb : m*mb);
    lapack::vector< std::complex<float> > work( lwork );

    LAPACK_ctpmlqt(
        &side_, &trans_, &m_, &n_, &k_, &l_, &mb_,
        (lapack_complex_float*) V, &ldv_,
        (lapack_complex_float*) T, &ldt_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Applies a complex orthogonal matrix Q obtained from a
/// "triangular-pentagonal" complex block reflector H to a general
/// complex matrix C, which consists of two blocks A and B, as follows:
///
/// - side = Left,  trans = NoTrans:   $Q C$
/// - side = Right, trans = NoTrans:   $C Q$
/// - side = Left,  trans = ConjTrans: $Q^H C$
/// - side = Right, trans = ConjTrans: $C Q^H$
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @since LAPACK 3.7.0
///
/// @param[in] side
///     - lapack::Side::Left:  apply $Q$ or $Q^H$ from the Left;
///     - lapack::Side::Right: apply $Q$ or $Q^H$ from the Right.
///
/// @param[in] trans
///     - lapack::Op::NoTrans:   No transpose,        apply $Q$;
///     - lapack::Op::Trans:     Transpose,           apply $Q^T$ (real only);
///     - lapack::Op::ConjTrans: Conjugate-transpose, apply $Q^H$.
///
/// @param[in] m
///     The number of rows of the matrix B. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix B. n >= 0.
///
/// @param[in] k
///     The number of elementary reflectors whose product defines
///     the matrix Q.
///
/// @param[in] l
///     The order of the trapezoidal part of V.
///     k >= l >= 0. See Further Details.
///
/// @param[in] mb
///     The block size used for the storage of T. k >= mb >= 1.
///     This must be the same value of mb used to generate T
///     in `lapack::tplqt`.
///
/// @param[in] V
///     The m-by-k matrix V, stored in an lda-by-k array.
///     The i-th row must contain the vector which defines the
///     elementary reflector H(i), for i = 1,2,...,k, as returned by
///     `lapack::tplqt` in B. See Further Details.
///
/// @param[in] ldv
///     The leading dimension of the array V.
///     If side = Left, ldv >= max(1,m);
///     if side = Right, ldv >= max(1,n).
///
/// @param[in] T
///     The mb-by-k matrix T, stored in an ldt-by-k array.
///     The upper triangular factors of the block reflectors
///     as returned by `lapack::tplqt`, stored as a mb-by-k matrix.
///
/// @param[in] ldt
///     The leading dimension of the array T. ldt >= mb.
///
/// @param[in,out] A
///     If side = Left,  the k-by-n matrix A, stored in an lda-by-n array;
///     if side = Right, the m-by-k matrix A, stored in an lda-by-k array.
///     On exit, A is overwritten by the corresponding block of
///     $Q C$ or $Q^H C$ or $C Q$ or $C Q^H$. See Further Details.
///
/// @param[in] lda
///     The leading dimension of the array A.
///     If side = Left,  lda >= max(1,k);
///     If side = Right, lda >= max(1,m).
///
/// @param[in,out] B
///     The m-by-n matrix B, stored in an ldb-by-n array.
///     On entry, the m-by-n matrix B.
///     On exit, B is overwritten by the corresponding block of
///     $Q C$ or $Q^H C$ or $C Q$ or $C Q^H$. See Further Details.
///
/// @param[in] ldb
///     The leading dimension of the array B.
///     ldb >= max(1,m).
///
/// @return = 0: successful exit
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The columns of the pentagonal matrix V contain the elementary reflectors
/// H(1), H(2), ..., H(k); V is composed of a rectangular block V1 and a
/// trapezoidal block V2:
/// \[
///     V = \begin{bmatrix} V1  &  V2 \end{bmatrix}.
/// \]
/// The size of the trapezoidal block V2 is determined by the parameter l,
/// where 0 <= l <= k; V2 is lower trapezoidal, consisting of the first l
/// rows of a k-by-k upper triangular matrix. If l=k, V2 is lower triangular;
/// if l=0, there is no trapezoidal block, hence V = V1 is rectangular.
///
/// If side = Left:
/// \[
///     C = \begin{bmatrix}
///             A
///         \\  B
///     \end{bmatrix},
/// \]
/// where A is k-by-n, B is m-by-n and V is k-by-m.
///
/// If side = Right:
/// \[
///     C = \begin{bmatrix} A  &  B \end{bmatrix},
/// \]
/// where A is m-by-k, B is m-by-n and V is k-by-n.
///
/// The unitary matrix Q is formed from V and T.
///
/// @ingroup tplqt
int64_t tpmlqt(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t k, int64_t l, int64_t mb,
    std::complex<double> const* V, int64_t ldv,
    std::complex<double> const* T, int64_t ldt,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb )
{
    char side_ = to_char( side );
    char trans_ = to_char( trans );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int k_ = to_lapack_int( k );
    lapack_int l_ = to_lapack_int( l );
    lapack_int mb_ = to_lapack_int( mb );
    lapack_int ldv_ = to_lapack_int( ldv );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    // allocate workspace
    int64_t lwork = (side == Side::Left ? n*mb : m*mb);
    lapack::vector< std::complex<double> > work( lwork );

    LAPACK_ztpmlqt(
        &side_, &trans_, &m_, &n_, &k_, &l_, &mb_,
        (lapack_complex_double*) V, &ldv_,
        (lapack_complex_double*) T, &ldt_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.7.0
