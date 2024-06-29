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
/// @ingroup gesv
int64_t gesvx(
    lapack::Factored fact, lapack::Op trans, int64_t n, int64_t nrhs,
    float* A, int64_t lda,
    float* AF, int64_t ldaf,
    int64_t* ipiv,
    lapack::Equed* equed,
    float* R,
    float* C,
    float* B, int64_t ldb,
    float* X, int64_t ldx,
    float* rcond,
    float* ferr,
    float* berr,
    float* rpivotgrowth )
{
    char fact_ = to_char( fact );
    char trans_ = to_char( trans );
    char equed_ = to_char( *equed );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldaf_ = to_lapack_int( ldaf );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (4*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_sgesvx(
        &fact_, &trans_, &n_, &nrhs_,
        A, &lda_,
        AF, &ldaf_,
        ipiv_ptr,
        &equed_,
        R,
        C,
        B, &ldb_,
        X, &ldx_, rcond,
        ferr,
        berr,
        &work[0],
        &iwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    from_string( std::string( 1, equed_ ), equed );
    *rpivotgrowth = work[0];
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesv
int64_t gesvx(
    lapack::Factored fact, lapack::Op trans, int64_t n, int64_t nrhs,
    double* A, int64_t lda,
    double* AF, int64_t ldaf,
    int64_t* ipiv,
    lapack::Equed* equed,
    double* R,
    double* C,
    double* B, int64_t ldb,
    double* X, int64_t ldx,
    double* rcond,
    double* ferr,
    double* berr,
    double* rpivotgrowth )
{
    char fact_ = to_char( fact );
    char trans_ = to_char( trans );
    char equed_ = to_char( *equed );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldaf_ = to_lapack_int( ldaf );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (4*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_dgesvx(
        &fact_, &trans_, &n_, &nrhs_,
        A, &lda_,
        AF, &ldaf_,
        ipiv_ptr,
        &equed_,
        R,
        C,
        B, &ldb_,
        X, &ldx_, rcond,
        ferr,
        berr,
        &work[0],
        &iwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    from_string( std::string( 1, equed_ ), equed );
    *rpivotgrowth = work[0];
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesv
int64_t gesvx(
    lapack::Factored fact, lapack::Op trans, int64_t n, int64_t nrhs,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* AF, int64_t ldaf,
    int64_t* ipiv,
    lapack::Equed* equed,
    float* R,
    float* C,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* X, int64_t ldx,
    float* rcond,
    float* ferr,
    float* berr,
    float* rpivotgrowth )
{
    char fact_ = to_char( fact );
    char trans_ = to_char( trans );
    char equed_ = to_char( *equed );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldaf_ = to_lapack_int( ldaf );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (2*n) );
    lapack::vector< float > rwork( (2*n) );

    LAPACK_cgesvx(
        &fact_, &trans_, &n_, &nrhs_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) AF, &ldaf_,
        ipiv_ptr,
        &equed_,
        R,
        C,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) X, &ldx_, rcond,
        ferr,
        berr,
        (lapack_complex_float*) &work[0],
        &rwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    from_string( std::string( 1, equed_ ), equed );
    *rpivotgrowth = rwork[0];
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// Uses the LU factorization to compute the solution to a
/// system of linear equations
/// \[
///     A X = B,
/// \]
/// where A is an n-by-n matrix and X and B are n-by-nrhs matrices.
///
/// Error bounds on the solution and a condition estimate are also
/// provided.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] fact
///     Whether or not the factored form of the matrix A is
///     supplied on entry, and if not, whether the matrix A should be
///     equilibrated before it is factored.
///     - lapack::Factored::Factored:
///         On entry, AF and ipiv contain the factored form of A.
///         If equed != None, the matrix A has been
///         equilibrated with scaling factors given by R and C.
///         A, AF, and ipiv are not modified.
///
///     - lapack::Factored::NotFactored:
///         The matrix A will be copied to AF and factored.
///
///     - lapack::Factored::Equilibrate:
///         The matrix A will be equilibrated if necessary, then
///         copied to AF and factored.
///
/// @param[in] trans
///     The form of the system of equations:
///     - lapack::Op::NoTrans:   $A   X = B$ (No transpose)
///     - lapack::Op::Trans:     $A^T X = B$ (Transpose)
///     - lapack::Op::ConjTrans: $A^H X = B$ (Conjugate transpose)
///
/// @param[in] n
///     The number of linear equations, i.e., the order of the
///     matrix A. n >= 0.
///
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrices B and X. nrhs >= 0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     On entry, the n-by-n matrix A.
///     - If fact = Factored and equed != None,
///     then A must have been equilibrated by the scaling
///     factors in R and/or C.
///
///     - A is not modified if fact = Factored or NotFactored,
///     or if fact = Equilibrate and equed = None on exit.
///
///     - On exit, if equed != None, A is scaled as follows:
///       - equed = Row:  $A := \text{diag}(R) \; A$
///       - equed = Col:  $A := A \; \text{diag}(C)$
///       - equed = Both: $A := \text{diag}(R) \; A \; \text{diag}(C).$
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[in,out] AF
///     The n-by-n matrix AF, stored in an ldaf-by-n array.
///     - If fact = Factored, then AF is an input argument and on entry
///     contains the factors L and U from the factorization
///     $A = P L U$ as computed by `lapack::getrf`.
///
///     - If equed != None, then
///     AF is the factored form of the equilibrated matrix A.
///
///     - If fact = NotFactored, then AF is an output argument and on exit
///     returns the factors L and U from the factorization $A = P L U$
///     of the original matrix A.
///
///     - If fact = Equilibrate, then AF is an output argument and on exit
///     returns the factors L and U from the factorization $A = P L U$
///     of the equilibrated matrix A (see the description of A for
///     the form of the equilibrated matrix).
///
/// @param[in] ldaf
///     The leading dimension of the array AF. ldaf >= max(1,n).
///
/// @param[in,out] ipiv
///     The vector ipiv of length n.
///     - If fact = Factored, then ipiv is an input argument and on entry
///     contains the pivot indices from the factorization $A = P L U$
///     as computed by `lapack::getrf`; row i of the matrix was interchanged
///     with row ipiv(i).
///
///     - If fact = NotFactored, then ipiv is an output argument and on exit
///     contains the pivot indices from the factorization $A = P L U$
///     of the original matrix A.
///
///     - If fact = Equilibrate, then ipiv is an output argument and on exit
///     contains the pivot indices from the factorization $A = P L U$
///     of the equilibrated matrix A.
///
/// @param[in,out] equed
///     The form of equilibration that was done:
///     - lapack::Equed::None:
///         No equilibration (always true if fact = NotFactored).
///     - lapack::Equed::Row:
///         Row equilibration, i.e., A has been premultiplied by diag(R).
///     - lapack::Equed::Col:
///         Column equilibration, i.e., A has been postmultiplied by diag(C).
///     - lapack::Equed::Both:
///         Both row and column equilibration, i.e.,
///         A has been replaced by $\text{diag}(R) \; A \; \text{diag}(C).$
///     \n
///     equed is an input argument if fact = Factored; otherwise, it is an
///     output argument.
///
/// @param[in,out] R
///     The vector R of length n.
///     The row scale factors for A.
///     - If equed = Row or Both, A is multiplied on the left by diag(R);
///     - if equed = None or Col, R is not accessed.
///
///     - If fact = Factored, R is an input argument;
///     - otherwise, R is an output argument.
///
///     - If fact = Factored and equed = Row or Both,
///     each element of R must be positive.
///
/// @param[in,out] C
///     The vector C of length n.
///     The column scale factors for A.
///     - If equed = Col or Both, A is multiplied on the right by diag(C);
///     - if equed = None or Row, C is not accessed.
///
///     - If fact = Factored, C is an input argument;
///     - otherwise, C is an output argument.
///
///     - If fact = Factored and equed = Col or Both,
///     each element of C must be positive.
///
/// @param[in,out] B
///     The n-by-nrhs matrix B, stored in an ldb-by-nrhs array.
///     On entry, the n-by-nrhs right hand side matrix B.
///     On exit,
///     - if equed = None, B is not modified;
///     - if trans = NoTrans and equed = Row or Both, B is overwritten by
///     $\text{diag}(R) \; B;$
///     - if trans = Trans or ConjTrans and equed = Col or Both, B is
///     overwritten by $\text{diag}(C) \; B.$
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,n).
///
/// @param[out] X
///     The n-by-nrhs matrix X, stored in an ldx-by-nrhs array.
///     If successful or return value = n+1, the n-by-nrhs solution matrix X
///     to the original system of equations. Note that A and B are
///     modified on exit if equed != None, and the solution to the
///     equilibrated system is $\text{diag}(C)^{-1} X$ if trans = NoTrans and
///     equed = Col or Both, or $\text{diag}(R)^{-1} X$ if trans = Trans or ConjTrans
///     and equed = Row or Both.
///
/// @param[in] ldx
///     The leading dimension of the array X. ldx >= max(1,n).
///
/// @param[out] rcond
///     The estimate of the reciprocal condition number of the matrix
///     A after equilibration (if done). If rcond is less than the
///     machine precision (in particular, if rcond = 0), the matrix
///     is singular to working precision. This condition is
///     indicated by a return code of return value > 0.
///
/// @param[out] ferr
///     The vector ferr of length nrhs.
///     The estimated forward error bound for each solution vector
///     X(j) (the j-th column of the solution matrix X).
///     If XTRUE is the true solution corresponding to X(j), ferr(j)
///     is an estimated upper bound for the magnitude of the largest
///     element in (X(j) - XTRUE) divided by the magnitude of the
///     largest element in X(j). The estimate is as reliable as
///     the estimate for rcond, and is almost always a slight
///     overestimate of the true error.
///
/// @param[out] berr
///     The vector berr of length nrhs.
///     The componentwise relative backward error of each solution
///     vector X(j) (i.e., the smallest relative change in
///     any element of A or B that makes X(j) an exact solution).
///
/// @param[out] rpivotgrowth
///     The reciprocal pivot growth
///     factor norm(A)/norm(U). The "max absolute element" norm is
///     used. If pivot growth is much less than 1, then the stability
///     of the LU factorization of the (equilibrated) matrix A
///     could be poor. This also means that the solution X, condition
///     estimator rcond, and forward error bound ferr could be
///     unreliable. If factorization fails with 0 < info <= n, then
///     rpivotgrowth contains the reciprocal pivot growth factor for the
///     leading info columns of A.
///
/// @return = 0: successful exit
/// @return > 0 and <= n: if return value = i,
///     then U(i,i) is exactly zero. The factorization has
///     been completed, but the factor U is exactly
///     singular, so the solution and error bounds
///     could not be computed. rcond = 0 is returned.
/// @return = n+1: U is nonsingular, but rcond is less than machine
///     precision, meaning that the matrix is singular
///     to working precision. Nevertheless, the
///     solution and error bounds are computed because
///     there are a number of situations where the
///     computed solution can be more accurate than the
///     value of rcond would suggest.
///
/// @ingroup gesv
int64_t gesvx(
    lapack::Factored fact, lapack::Op trans, int64_t n, int64_t nrhs,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* AF, int64_t ldaf,
    int64_t* ipiv,
    lapack::Equed* equed,
    double* R,
    double* C,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* X, int64_t ldx,
    double* rcond,
    double* ferr,
    double* berr,
    double* rpivotgrowth )
{
    char fact_ = to_char( fact );
    char trans_ = to_char( trans );
    char equed_ = to_char( *equed );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldaf_ = to_lapack_int( ldaf );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (2*n) );
    lapack::vector< double > rwork( (2*n) );

    LAPACK_zgesvx(
        &fact_, &trans_, &n_, &nrhs_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) AF, &ldaf_,
        ipiv_ptr,
        &equed_,
        R,
        C,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) X, &ldx_, rcond,
        ferr,
        berr,
        (lapack_complex_double*) &work[0],
        &rwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    from_string( std::string( 1, equed_ ), equed );
    *rpivotgrowth = rwork[0];
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

}  // namespace lapack
