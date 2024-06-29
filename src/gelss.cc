// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gels
int64_t gelss(
    int64_t m, int64_t n, int64_t nrhs,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    float* S, float rcond,
    int64_t* rank )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int rank_ = to_lapack_int( *rank );
    lapack_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_sgelss(
        &m_, &n_, &nrhs_,
        A, &lda_,
        B, &ldb_,
        S, &rcond, &rank_,
        qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< float > work( lwork_ );

    LAPACK_sgelss(
        &m_, &n_, &nrhs_,
        A, &lda_,
        B, &ldb_,
        S, &rcond, &rank_,
        &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *rank = rank_;
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gels
int64_t gelss(
    int64_t m, int64_t n, int64_t nrhs,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    double* S, double rcond,
    int64_t* rank )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int rank_ = to_lapack_int( *rank );
    lapack_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_dgelss(
        &m_, &n_, &nrhs_,
        A, &lda_,
        B, &ldb_,
        S, &rcond, &rank_,
        qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< double > work( lwork_ );

    LAPACK_dgelss(
        &m_, &n_, &nrhs_,
        A, &lda_,
        B, &ldb_,
        S, &rcond, &rank_,
        &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *rank = rank_;
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gels
int64_t gelss(
    int64_t m, int64_t n, int64_t nrhs,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    float* S, float rcond,
    int64_t* rank )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int rank_ = to_lapack_int( *rank );
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_cgelss(
        &m_, &n_, &nrhs_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_,
        S, &rcond, &rank_,
        (lapack_complex_float*) qry_work, &ineg_one,
        qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );
    lapack::vector< float > rwork( (5*min(m,n)) );

    LAPACK_cgelss(
        &m_, &n_, &nrhs_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_,
        S, &rcond, &rank_,
        (lapack_complex_float*) &work[0], &lwork_,
        &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *rank = rank_;
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the minimum norm solution to a complex linear
/// least squares problem:
///     minimize $|| b - A x ||_2$
/// using the singular value decomposition (SVD) of A. A is an m-by-n
/// matrix which may be rank-deficient.
///
/// Several right hand side vectors b and solution vectors x can be
/// handled in a single call; they are stored as the columns of the
/// m-by-nrhs right hand side matrix B and the n-by-nrhs solution matrix
/// X.
///
/// The effective rank of A is determined by treating as zero those
/// singular values which are less than rcond times the largest singular
/// value.
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
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrices B and X. nrhs >= 0.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     On entry, the m-by-n matrix A.
///     On exit, the first min(m,n) rows of A are overwritten with
///     its right singular vectors, stored rowwise.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @param[in,out] B
///     The max(m,n)-by-nrhs matrix B or X, stored in an ldb-by-nrhs array.
///     On entry, the m-by-nrhs right hand side matrix B.
///     On exit, B is overwritten by the n-by-nrhs solution matrix X.
///     If m >= n and rank = n, the residual sum-of-squares for
///     the solution in the i-th column is given by the sum of
///     squares of the modulus of elements n+1:m in that column.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,m,n).
///
/// @param[out] S
///     The vector S of length min(m,n).
///     The singular values of A in decreasing order.
///     The condition number of A in the 2-norm = S(1)/S(min(m,n)).
///
/// @param[in] rcond
///     rcond is used to determine the effective rank of A.
///     Singular values S(i) <= rcond*S(1) are treated as zero.
///     If rcond < 0, machine precision is used instead.
///
/// @param[out] rank
///     The effective rank of A, i.e., the number of singular values
///     which are greater than rcond*S(1).
///
/// @return = 0: successful exit
/// @return > 0: the algorithm for computing the SVD failed to converge;
///     if return value = i, i off-diagonal elements of an intermediate
///     bidiagonal form did not converge to zero.
///
/// @ingroup gels
int64_t gelss(
    int64_t m, int64_t n, int64_t nrhs,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    double* S, double rcond,
    int64_t* rank )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int rank_ = to_lapack_int( *rank );
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_zgelss(
        &m_, &n_, &nrhs_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_,
        S, &rcond, &rank_,
        (lapack_complex_double*) qry_work, &ineg_one,
        qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );
    lapack::vector< double > rwork( (5*min(m,n)) );

    LAPACK_zgelss(
        &m_, &n_, &nrhs_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_,
        S, &rcond, &rank_,
        (lapack_complex_double*) &work[0], &lwork_,
        &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *rank = rank_;
    return info_;
}

}  // namespace lapack
