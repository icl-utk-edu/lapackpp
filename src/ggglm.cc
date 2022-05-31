// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup ggls
int64_t ggglm(
    int64_t n, int64_t m, int64_t p,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    float* D,
    float* X,
    float* Y )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int m_ = (lapack_int) m;
    lapack_int p_ = (lapack_int) p;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_sggglm(
        &n_, &m_, &p_,
        A, &lda_,
        B, &ldb_,
        D,
        X,
        Y,
        qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< float > work( lwork_ );

    LAPACK_sggglm(
        &n_, &m_, &p_,
        A, &lda_,
        B, &ldb_,
        D,
        X,
        Y,
        &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ggls
int64_t ggglm(
    int64_t n, int64_t m, int64_t p,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    double* D,
    double* X,
    double* Y )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int m_ = (lapack_int) m;
    lapack_int p_ = (lapack_int) p;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_dggglm(
        &n_, &m_, &p_,
        A, &lda_,
        B, &ldb_,
        D,
        X,
        Y,
        qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< double > work( lwork_ );

    LAPACK_dggglm(
        &n_, &m_, &p_,
        A, &lda_,
        B, &ldb_,
        D,
        X,
        Y,
        &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ggls
int64_t ggglm(
    int64_t n, int64_t m, int64_t p,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* D,
    std::complex<float>* X,
    std::complex<float>* Y )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int m_ = (lapack_int) m;
    lapack_int p_ = (lapack_int) p;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_cggglm(
        &n_, &m_, &p_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) D,
        (lapack_complex_float*) X,
        (lapack_complex_float*) Y,
        (lapack_complex_float*) qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );

    LAPACK_cggglm(
        &n_, &m_, &p_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) D,
        (lapack_complex_float*) X,
        (lapack_complex_float*) Y,
        (lapack_complex_float*) &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Solves a general Gauss-Markov linear model (GLM) problem:
/// \[
///     \min_x || y ||_2  \text{ subject to } d = A x + B y,
/// \]
/// where A is an n-by-m matrix, B is an n-by-p matrix, and d is a
/// given n-vector. It is assumed that m <= n <= m+p, and
/// \[
///     rank(A) = m \text{ and } rank( [A, B] ) = n.
/// \]
/// Under these assumptions, the constrained equation is always
/// consistent, and there is a unique solution x and a minimal 2-norm
/// solution y, which is obtained using a generalized QR factorization
/// of the matrices (A, B) given by
/// \[
///     A = Q \begin{bmatrix}
///            R
///         \\ 0
///     \end{bmatrix},
///     \quad
///     B = Q T Z.
/// \]
///
/// In particular, if matrix B is square nonsingular, then the problem
/// GLM is equivalent to the following weighted linear least squares
/// problem
/// \[
///     \min_x || B^{-1} (d - A x) ||_2.
/// \]
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] n
///     The number of rows of the matrices A and B. n >= 0.
///
/// @param[in] m
///     The number of columns of the matrix A. 0 <= m <= n.
///
/// @param[in] p
///     The number of columns of the matrix B. p >= n-m.
///
/// @param[in,out] A
///     The n-by-m matrix A, stored in an lda-by-m array.
///     On entry, the n-by-m matrix A.
///     On exit, the upper triangular part of the array A contains
///     the m-by-m upper triangular matrix R.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[in,out] B
///     The n-by-p matrix B, stored in an ldb-by-p array.
///     On entry, the n-by-p matrix B.
///     On exit, if n <= p, the upper triangle of the subarray
///     B(1:n,p-n+1:p) contains the n-by-n upper triangular matrix T;
///     if n > p, the elements on and above the (n-p)th subdiagonal
///     contain the n-by-p upper trapezoidal matrix T.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,n).
///
/// @param[in,out] D
///     The vector D of length n.
///     On entry, D is the left hand side of the GLM equation.
///     On exit, D is destroyed.
///
/// @param[out] X
///     The vector X of length m.
///
/// @param[out] Y
///     The vector Y of length p.
///     On exit, X and Y are the solutions of the GLM problem.
///
/// @return = 0: successful exit.
/// @return = 1: the upper triangular factor R associated with A in the
///     generalized QR factorization of the pair (A, B) is
///     singular, so that rank(A) < m; the least squares
///     solution could not be computed.
/// @return = 2: the bottom (n-m) by (n-m) part of the upper trapezoidal
///     factor T associated with B in the generalized QR
///     factorization of the pair (A, B) is singular, so that
///     $rank( [A, B] ) < n$;
///     the least squares solution could not be computed.
///
/// @ingroup ggls
int64_t ggglm(
    int64_t n, int64_t m, int64_t p,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* D,
    std::complex<double>* X,
    std::complex<double>* Y )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int m_ = (lapack_int) m;
    lapack_int p_ = (lapack_int) p;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_zggglm(
        &n_, &m_, &p_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) D,
        (lapack_complex_double*) X,
        (lapack_complex_double*) Y,
        (lapack_complex_double*) qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );

    LAPACK_zggglm(
        &n_, &m_, &p_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) D,
        (lapack_complex_double*) X,
        (lapack_complex_double*) Y,
        (lapack_complex_double*) &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
