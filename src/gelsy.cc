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
int64_t gelsy(
    int64_t m, int64_t n, int64_t nrhs,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    int64_t* jpvt, float rcond,
    int64_t* rank )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > jpvt_( &jpvt[0], &jpvt[(n)] );
        lapack_int* jpvt_ptr = &jpvt_[0];
    #else
        lapack_int* jpvt_ptr = jpvt;
    #endif
    lapack_int rank_ = 0;
    lapack_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_sgelsy(
        &m_, &n_, &nrhs_,
        A, &lda_,
        B, &ldb_,
        jpvt_ptr, &rcond, &rank_,
        qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< float > work( lwork_ );

    LAPACK_sgelsy(
        &m_, &n_, &nrhs_,
        A, &lda_,
        B, &ldb_,
        jpvt_ptr, &rcond, &rank_,
        &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( jpvt_.begin(), jpvt_.end(), jpvt );
    #endif
    *rank = rank_;
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gels
int64_t gelsy(
    int64_t m, int64_t n, int64_t nrhs,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    int64_t* jpvt, double rcond,
    int64_t* rank )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > jpvt_( &jpvt[0], &jpvt[(n)] );
        lapack_int* jpvt_ptr = &jpvt_[0];
    #else
        lapack_int* jpvt_ptr = jpvt;
    #endif
    lapack_int rank_ = 0;
    lapack_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_dgelsy(
        &m_, &n_, &nrhs_,
        A, &lda_,
        B, &ldb_,
        jpvt_ptr, &rcond, &rank_,
        qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< double > work( lwork_ );

    LAPACK_dgelsy(
        &m_, &n_, &nrhs_,
        A, &lda_,
        B, &ldb_,
        jpvt_ptr, &rcond, &rank_,
        &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( jpvt_.begin(), jpvt_.end(), jpvt );
    #endif
    *rank = rank_;
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gels
int64_t gelsy(
    int64_t m, int64_t n, int64_t nrhs,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    int64_t* jpvt, float rcond,
    int64_t* rank )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > jpvt_( &jpvt[0], &jpvt[(n)] );
        lapack_int* jpvt_ptr = &jpvt_[0];
    #else
        lapack_int* jpvt_ptr = jpvt;
    #endif
    lapack_int rank_ = 0;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_cgelsy(
        &m_, &n_, &nrhs_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_,
        jpvt_ptr, &rcond, &rank_,
        (lapack_complex_float*) qry_work, &ineg_one,
        qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );
    lapack::vector< float > rwork( (2*n) );

    LAPACK_cgelsy(
        &m_, &n_, &nrhs_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_,
        jpvt_ptr, &rcond, &rank_,
        (lapack_complex_float*) &work[0], &lwork_,
        &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( jpvt_.begin(), jpvt_.end(), jpvt );
    #endif
    *rank = rank_;
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the minimum-norm solution to a complex linear least
/// squares problem:
///     minimize $|| A X - B ||_2$
/// using a complete orthogonal factorization of A. A is an m-by-n
/// matrix which may be rank-deficient.
///
/// Several right hand side vectors b and solution vectors x can be
/// handled in a single call; they are stored as the columns of the
/// m-by-nrhs right hand side matrix B and the n-by-nrhs solution
/// matrix X.
///
/// The routine first computes a QR factorization with column pivoting:
/// \[
///     A P = Q \begin{bmatrix}
///             R_{11}  &  R_{12}
///         \\  0       &  R_{22}
///     \end{bmatrix},
/// \]
/// with R11 defined as the largest leading submatrix whose estimated
/// condition number is less than 1/rcond. The order of R11, rank,
/// is the effective rank of A.
///
/// Then, R22 is considered to be negligible, and R12 is annihilated
/// by unitary transformations from the right, arriving at the
/// complete orthogonal factorization:
/// \[
///     A P = Q \begin{bmatrix}
///             T_{11}  &  0
///         \\  0       &  0
///     \end{bmatrix} Z.
/// \]
///
/// The minimum-norm solution is then
/// \[
///     X = P Z^H \begin{bmatrix}
///             T_{11}^{-1} Q_1^H B
///         \\  0
///     \end{bmatrix},
/// \]
/// where $Q_1$ consists of the first rank columns of Q.
///
/// This routine is basically identical to the original gelsx except
/// three differences:
///   - The permutation of matrix B (the right hand side) is faster and simpler.
///   - The call to the subroutine geqpf has been substituted by the
///     the call to the subroutine geqp3. This subroutine is a BLAS-3
///     version of the QR factorization with column pivoting.
///   - Matrix B (the right hand side) is updated with BLAS-3.
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
///     The number of right hand sides, i.e., the number of
///     columns of matrices B and X. nrhs >= 0.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     On entry, the m-by-n matrix A.
///     On exit, A has been overwritten by details of its
///     complete orthogonal factorization.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @param[in,out] B
///     The max(m,n)-by-nrhs matrix B or X, stored in an ldb-by-nrhs array.
///     On entry, the m-by-nrhs right hand side matrix B.
///     On exit, the n-by-nrhs solution matrix X.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,m,n).
///
/// @param[in,out] jpvt
///     The vector jpvt of length n.
///     On entry, if jpvt(i) != 0, the i-th column of A is permuted
///     to the front of AP, otherwise column i is a free column.
///     On exit, if jpvt(i) = k, then the i-th column of $A P$
///     was the k-th column of A.
///
/// @param[in] rcond
///     rcond is used to determine the effective rank of A, which
///     is defined as the order of the largest leading triangular
///     submatrix R11 in the QR factorization with pivoting of A,
///     whose estimated condition number < 1/rcond.
///
/// @param[out] rank
///     The effective rank of A, i.e., the order of the submatrix
///     R11. This is the same as the order of the submatrix T11
///     in the complete orthogonal factorization of A.
///
/// @return = 0: successful exit
///
/// @ingroup gels
int64_t gelsy(
    int64_t m, int64_t n, int64_t nrhs,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    int64_t* jpvt, double rcond,
    int64_t* rank )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > jpvt_( &jpvt[0], &jpvt[(n)] );
        lapack_int* jpvt_ptr = &jpvt_[0];
    #else
        lapack_int* jpvt_ptr = jpvt;
    #endif
    lapack_int rank_ = 0;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_zgelsy(
        &m_, &n_, &nrhs_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_,
        jpvt_ptr, &rcond, &rank_,
        (lapack_complex_double*) qry_work, &ineg_one,
        qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );
    lapack::vector< double > rwork( (2*n) );

    LAPACK_zgelsy(
        &m_, &n_, &nrhs_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_,
        jpvt_ptr, &rcond, &rank_,
        (lapack_complex_double*) &work[0], &lwork_,
        &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( jpvt_.begin(), jpvt_.end(), jpvt );
    #endif
    *rank = rank_;
    return info_;
}

}  // namespace lapack
