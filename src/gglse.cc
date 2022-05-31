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
int64_t gglse(
    int64_t m, int64_t n, int64_t p,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    float* C,
    float* D,
    float* X )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int p_ = (lapack_int) p;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_sgglse(
        &m_, &n_, &p_,
        A, &lda_,
        B, &ldb_,
        C,
        D,
        X,
        qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< float > work( lwork_ );

    LAPACK_sgglse(
        &m_, &n_, &p_,
        A, &lda_,
        B, &ldb_,
        C,
        D,
        X,
        &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ggls
int64_t gglse(
    int64_t m, int64_t n, int64_t p,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    double* C,
    double* D,
    double* X )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int p_ = (lapack_int) p;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_dgglse(
        &m_, &n_, &p_,
        A, &lda_,
        B, &ldb_,
        C,
        D,
        X,
        qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< double > work( lwork_ );

    LAPACK_dgglse(
        &m_, &n_, &p_,
        A, &lda_,
        B, &ldb_,
        C,
        D,
        X,
        &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ggls
int64_t gglse(
    int64_t m, int64_t n, int64_t p,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* C,
    std::complex<float>* D,
    std::complex<float>* X )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int p_ = (lapack_int) p;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_cgglse(
        &m_, &n_, &p_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) C,
        (lapack_complex_float*) D,
        (lapack_complex_float*) X,
        (lapack_complex_float*) qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );

    LAPACK_cgglse(
        &m_, &n_, &p_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) C,
        (lapack_complex_float*) D,
        (lapack_complex_float*) X,
        (lapack_complex_float*) &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Solves the linear equality-constrained least squares (LSE)
/// problem:
/// \[
///     \min_x || c - A x ||_2 \text{ subject to } B x = d
/// \]
///
/// where A is an m-by-n matrix, B is a p-by-n matrix, c is a given
/// m-vector, and d is a given p-vector. It is assumed that
/// p <= n <= m+p, and
/// \[
///     rank(B) = p
/// \]
/// and
/// \[
///     rank\left( \begin{bmatrix}
///            A
///         \\ B
///     \end{bmatrix} \right) = n.
/// \]
///
/// These conditions ensure that the LSE problem has a unique solution,
/// which is obtained using a generalized RQ factorization of the
/// matrices (B, A) given by
/// \[
///     B = \begin{bmatrix} 0  &  R \end{bmatrix} Q, \quad A = Z T Q.
/// \]
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] m
///     The number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrices A and B. n >= 0.
///
/// @param[in] p
///     The number of rows of the matrix B. 0 <= p <= n <= m+p.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     On entry, the m-by-n matrix A.
///     On exit, the elements on and above the diagonal of the array
///     contain the min(m,n)-by-n upper trapezoidal matrix T.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @param[in,out] B
///     The p-by-n matrix B, stored in an ldb-by-n array.
///     On entry, the p-by-n matrix B.
///     On exit, the upper triangle of the subarray B(1:p,n-p+1:n)
///     contains the p-by-p upper triangular matrix R.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,p).
///
/// @param[in,out] C
///     The vector C of length m.
///     On entry, C contains the right hand side vector for the
///     least squares part of the LSE problem.
///     On exit, the residual sum of squares for the solution
///     is given by the sum of squares of elements n-p+1 to m of
///     vector C.
///
/// @param[in,out] D
///     The vector D of length p.
///     On entry, D contains the right hand side vector for the
///     constrained equation.
///     On exit, D is destroyed.
///
/// @param[out] X
///     The vector X of length n.
///     On exit, X is the solution of the LSE problem.
///
/// @return = 0: successful exit.
/// @return = 1: the upper triangular factor R associated with B in the
///     generalized RQ factorization of the pair (B, A) is
///     singular, so that rank(B) < p; the least squares
///     solution could not be computed.
/// @return = 2: the (n-p) by (n-p) part of the upper trapezoidal factor
///     T associated with A in the generalized RQ factorization
///     of the pair (B, A) is singular, so that
///     \[
///     rank\left( \begin{bmatrix}
///            A
///         \\ B
///     \end{bmatrix} \right) < n;
///     \]
///     the least squares solution could not be computed.
///
/// @ingroup ggls
int64_t gglse(
    int64_t m, int64_t n, int64_t p,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* C,
    std::complex<double>* D,
    std::complex<double>* X )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int p_ = (lapack_int) p;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_zgglse(
        &m_, &n_, &p_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) C,
        (lapack_complex_double*) D,
        (lapack_complex_double*) X,
        (lapack_complex_double*) qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );

    LAPACK_zgglse(
        &m_, &n_, &p_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) C,
        (lapack_complex_double*) D,
        (lapack_complex_double*) X,
        (lapack_complex_double*) &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
