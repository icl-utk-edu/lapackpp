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
/// @ingroup ppsv
int64_t ppsvx(
    lapack::Factored fact, lapack::Uplo uplo, int64_t n, int64_t nrhs,
    float* AP,
    float* AFP,
    lapack::Equed* equed,
    float* S,
    float* B, int64_t ldb,
    float* X, int64_t ldx,
    float* rcond,
    float* ferr,
    float* berr )
{
    char fact_ = to_char( fact );
    char uplo_ = to_char( uplo );
    char equed_ = to_char( *equed );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (3*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_sppsvx(
        &fact_, &uplo_, &n_, &nrhs_,
        AP,
        AFP,
        &equed_,
        S,
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
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ppsv
int64_t ppsvx(
    lapack::Factored fact, lapack::Uplo uplo, int64_t n, int64_t nrhs,
    double* AP,
    double* AFP,
    lapack::Equed* equed,
    double* S,
    double* B, int64_t ldb,
    double* X, int64_t ldx,
    double* rcond,
    double* ferr,
    double* berr )
{
    char fact_ = to_char( fact );
    char uplo_ = to_char( uplo );
    char equed_ = to_char( *equed );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (3*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_dppsvx(
        &fact_, &uplo_, &n_, &nrhs_,
        AP,
        AFP,
        &equed_,
        S,
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
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ppsv
int64_t ppsvx(
    lapack::Factored fact, lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<float>* AP,
    std::complex<float>* AFP,
    lapack::Equed* equed,
    float* S,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* X, int64_t ldx,
    float* rcond,
    float* ferr,
    float* berr )
{
    char fact_ = to_char( fact );
    char uplo_ = to_char( uplo );
    char equed_ = to_char( *equed );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (2*n) );
    lapack::vector< float > rwork( (n) );

    LAPACK_cppsvx(
        &fact_, &uplo_, &n_, &nrhs_,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) AFP,
        &equed_,
        S,
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
    return info_;
}

// -----------------------------------------------------------------------------
/// Uses the Cholesky factorization
///     $A = U^H U$ or
///     $A = L L^H$ to
/// compute the solution to a system of linear equations
/// \[
///     A X = B,
/// \]
/// where A is an n-by-n Hermitian positive definite matrix stored in
/// packed format and X and B are n-by-nrhs matrices.
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
///         On entry, AFP contains the factored form of A.
///         If equed = Yes, the matrix A has been equilibrated
///         with scaling factors given by S.
///         AP and AFP will not be modified.
///
///     - lapack::Factored::NotFactored:
///         The matrix A will be copied to AFP and factored.
///
///     - lapack::Factored::Equilibrate:
///         The matrix A will be equilibrated if necessary, then
///         copied to AFP and factored.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangle of A is stored;
///     - lapack::Uplo::Lower: Lower triangle of A is stored.
///
/// @param[in] n
///     The number of linear equations, i.e., the order of the
///     matrix A. n >= 0.
///
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrices B and X. nrhs >= 0.
///
/// @param[in,out] AP
///     The vector AP of length n*(n+1)/2.
///     - On entry, the upper or lower triangle of the Hermitian matrix
///     A, packed columnwise in a linear array, except if fact = Factored
///     and equed = Yes, then A must contain the equilibrated matrix
///     $\text{diag}(S) \; A \; \text{diag}(S)$.
///     The j-th column of A is stored in the array AP as follows:
///       - if uplo = Upper, AP(i + (j-1)*j/2) = A(i,j) for 1 <= i <= j;
///       - if uplo = Lower, AP(i + (j-1)*(2n-j)/2) = A(i,j) for j <= i <= n.
///         \n
///         See below for further details.
///
///     - A is not modified if fact = Factored or NotFactored,
///     or if fact = Equilibrate and equed = None on exit.
///
///     - On exit, if fact = Equilibrate and equed = Yes, A is overwritten by
///     $\text{diag}(S) \; A \; \text{diag}(S)$.
///
/// @param[in,out] AFP
///     The vector AFP of length n*(n+1)/2.
///     - If fact = Factored, then AFP is an input argument and on entry
///     contains the triangular factor U or L from the Cholesky
///     factorization $A = U^H U$ or $A = L L^H,$ in the same storage
///     format as A.
///
///     - If equed != None, then AFP is the factored
///     form of the equilibrated matrix A.
///
///     - If fact = NotFactored, then AFP is an output argument and on exit
///     returns the triangular factor U or L from the Cholesky
///     factorization $A = U^H U$ or $A = L L^H$ of the original
///     matrix A.
///
///     - If fact = Equilibrate, then AFP is an output argument and on exit
///     returns the triangular factor U or L from the Cholesky
///     factorization $A = U^H U$ or $A = L L^H$ of the equilibrated
///     matrix A (see the description of AP for the form of the
///     equilibrated matrix).
///
/// @param[in,out] equed
///     The form of equilibration that was done:
///     - lapack::Equed::None:
///         No equilibration (always true if fact = NotFactored).
///     - lapack::Equed::Yes:
///         Equilibration was done, i.e.,
///         A has been replaced by $\text{diag}(S) \; A \; \text{diag}(S).$
///
///     - equed is an input argument if fact = Factored; otherwise, it is an
///     output argument.
///
/// @param[in,out] S
///     The vector S of length n.
///     The scale factors for A.
///     - If fact = Factored, S is an input argument;
///     - otherwise, S is an output argument.
///     - If fact = Factored and equed = Yes, each element of S must be positive.
///     - If equed = None, S is not accessed.
///
/// @param[in,out] B
///     The n-by-nrhs matrix B, stored in an ldb-by-nrhs array.
///     On entry, the n-by-nrhs right hand side matrix B.
///     On exit, if equed = None, B is not modified; if equed = Yes,
///     B is overwritten by $\text{diag}(S) \; B.$
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,n).
///
/// @param[out] X
///     The n-by-nrhs matrix X, stored in an ldx-by-nrhs array.
///     If successful or return value = n+1, the n-by-nrhs solution matrix X to
///     the original system of equations. Note that if equed = Yes,
///     A and B are modified on exit, and the solution to the
///     equilibrated system is $\text{diag}(S)^{-1} X.$
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
/// @return = 0: successful exit
/// @return > 0 and <= n: if return value = i,
///     the leading minor of order i of A is
///     not positive definite, so the factorization
///     could not be completed, and the solution has not
///     been computed. rcond = 0 is returned.
/// @return = n+1: U is nonsingular, but rcond is less than machine
///     precision, meaning that the matrix is singular
///     to working precision. Nevertheless, the
///     solution and error bounds are computed because
///     there are a number of situations where the
///     computed solution can be more accurate than the
///     value of rcond would suggest.
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The packed storage scheme is illustrated by the following example
/// when n = 4, uplo = Upper:
///
/// Two-dimensional storage of the Hermitian matrix A:
///
///     [ a11 a12 a13 a14 ]
///     [     a22 a23 a24 ]
///     [         a33 a34 ]    (aij = conj(aji))
///     [             a44 ]
///
/// Packed storage of the upper triangle of A:
///
///     AP = [ a11, a12, a22, a13, a23, a33, a14, a24, a34, a44 ]
///
/// @ingroup ppsv
int64_t ppsvx(
    lapack::Factored fact, lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<double>* AP,
    std::complex<double>* AFP,
    lapack::Equed* equed,
    double* S,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* X, int64_t ldx,
    double* rcond,
    double* ferr,
    double* berr )
{
    char fact_ = to_char( fact );
    char uplo_ = to_char( uplo );
    char equed_ = to_char( *equed );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (2*n) );
    lapack::vector< double > rwork( (n) );

    LAPACK_zppsvx(
        &fact_, &uplo_, &n_, &nrhs_,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) AFP,
        &equed_,
        S,
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
    return info_;
}

}  // namespace lapack
