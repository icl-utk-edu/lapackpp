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
/// @ingroup ptsv
int64_t ptsvx(
    lapack::Factored fact, int64_t n, int64_t nrhs,
    float const* D,
    float const* E,
    float* DF,
    float* EF,
    float const* B, int64_t ldb,
    float* X, int64_t ldx,
    float* rcond,
    float* ferr,
    float* berr )
{
    char fact_ = to_char( fact );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (2*n) );

    LAPACK_sptsvx(
        &fact_, &n_, &nrhs_,
        D,
        E,
        DF,
        EF,
        B, &ldb_,
        X, &ldx_, rcond,
        ferr,
        berr,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ptsv
int64_t ptsvx(
    lapack::Factored fact, int64_t n, int64_t nrhs,
    double const* D,
    double const* E,
    double* DF,
    double* EF,
    double const* B, int64_t ldb,
    double* X, int64_t ldx,
    double* rcond,
    double* ferr,
    double* berr )
{
    char fact_ = to_char( fact );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (2*n) );

    LAPACK_dptsvx(
        &fact_, &n_, &nrhs_,
        D,
        E,
        DF,
        EF,
        B, &ldb_,
        X, &ldx_, rcond,
        ferr,
        berr,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ptsv
int64_t ptsvx(
    lapack::Factored fact, int64_t n, int64_t nrhs,
    float const* D,
    std::complex<float> const* E,
    float* DF,
    std::complex<float>* EF,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float>* X, int64_t ldx,
    float* rcond,
    float* ferr,
    float* berr )
{
    char fact_ = to_char( fact );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (n) );
    lapack::vector< float > rwork( (n) );

    LAPACK_cptsvx(
        &fact_, &n_, &nrhs_,
        D,
        (lapack_complex_float*) E,
        DF,
        (lapack_complex_float*) EF,
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
    return info_;
}

// -----------------------------------------------------------------------------
/// Uses the factorization $A = L D L^H$ to compute the solution
/// to a system of linear equations $A X = B,$ where A is an
/// n-by-n Hermitian positive definite tridiagonal matrix and X and B
/// are n-by-nrhs matrices.
///
/// Error bounds on the solution and a condition estimate are also
/// provided.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] fact
///     Specifies whether or not the factored form of the matrix
///     A is supplied on entry.
///     - lapack::Factored::Factored: On entry, DF and EF contain the factored form of A.
///         D, E, DF, and EF will not be modified.
///     - lapack::Factored::NotFactored: The matrix A will be copied to DF and EF and
///         factored.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrices B and X. nrhs >= 0.
///
/// @param[in] D
///     The vector D of length n.
///     The n diagonal elements of the tridiagonal matrix A.
///
/// @param[in] E
///     The vector E of length n-1.
///     The (n-1) subdiagonal elements of the tridiagonal matrix A.
///
/// @param[in,out] DF
///     The vector DF of length n.
///     - If fact = Factored, then DF is an input argument and on entry
///     contains the n diagonal elements of the diagonal matrix D
///     from the $L D L^H$ factorization of A.
///
///     - If fact = NotFactored, then DF is an output argument and on exit
///     contains the n diagonal elements of the diagonal matrix D
///     from the $L D L^H$ factorization of A.
///
/// @param[in,out] EF
///     The vector EF of length n-1.
///     - If fact = Factored, then EF is an input argument and on entry
///     contains the (n-1) subdiagonal elements of the unit
///     bidiagonal factor L from the $L D L^H$ factorization of A.
///
///     - If fact = NotFactored, then EF is an output argument and on exit
///     contains the (n-1) subdiagonal elements of the unit
///     bidiagonal factor L from the $L D L^H$ factorization of A.
///
/// @param[in] B
///     The n-by-nrhs matrix B, stored in an ldb-by-nrhs array.
///     The n-by-nrhs right hand side matrix B.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,n).
///
/// @param[out] X
///     The n-by-nrhs matrix X, stored in an ldx-by-nrhs array.
///     If successful or return value = n+1, the n-by-nrhs solution matrix X.
///
/// @param[in] ldx
///     The leading dimension of the array X. ldx >= max(1,n).
///
/// @param[out] rcond
///     The reciprocal condition number of the matrix A. If rcond
///     is less than the machine precision (in particular, if
///     rcond = 0), the matrix is singular to working precision.
///     This condition is indicated by a return code of return value > 0.
///
/// @param[out] ferr
///     The vector ferr of length nrhs.
///     The forward error bound for each solution vector
///     X(j) (the j-th column of the solution matrix X).
///     If XTRUE is the true solution corresponding to X(j), ferr(j)
///     is an estimated upper bound for the magnitude of the largest
///     element in (X(j) - XTRUE) divided by the magnitude of the
///     largest element in X(j).
///
/// @param[out] berr
///     The vector berr of length nrhs.
///     The componentwise relative backward error of each solution
///     vector X(j) (i.e., the smallest relative change in any
///     element of A or B that makes X(j) an exact solution).
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
/// @ingroup ptsv
int64_t ptsvx(
    lapack::Factored fact, int64_t n, int64_t nrhs,
    double const* D,
    std::complex<double> const* E,
    double* DF,
    std::complex<double>* EF,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double>* X, int64_t ldx,
    double* rcond,
    double* ferr,
    double* berr )
{
    char fact_ = to_char( fact );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (n) );
    lapack::vector< double > rwork( (n) );

    LAPACK_zptsvx(
        &fact_, &n_, &nrhs_,
        D,
        (lapack_complex_double*) E,
        DF,
        (lapack_complex_double*) EF,
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
    return info_;
}

}  // namespace lapack
