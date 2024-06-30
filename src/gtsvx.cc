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
/// @ingroup gtsv
int64_t gtsvx(
    lapack::Factored fact, lapack::Op trans, int64_t n, int64_t nrhs,
    float const* DL,
    float const* D,
    float const* DU,
    float* DLF,
    float* DF,
    float* DUF,
    float* DU2,
    int64_t* ipiv,
    float const* B, int64_t ldb,
    float* X, int64_t ldx,
    float* rcond,
    float* ferr,
    float* berr )
{
    char fact_ = to_char( fact );
    char trans_ = to_char( trans );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
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
    lapack::vector< float > work( (3*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_sgtsvx(
        &fact_, &trans_, &n_, &nrhs_,
        DL,
        D,
        DU,
        DLF,
        DF,
        DUF,
        DU2,
        ipiv_ptr,
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
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gtsv
int64_t gtsvx(
    lapack::Factored fact, lapack::Op trans, int64_t n, int64_t nrhs,
    double const* DL,
    double const* D,
    double const* DU,
    double* DLF,
    double* DF,
    double* DUF,
    double* DU2,
    int64_t* ipiv,
    double const* B, int64_t ldb,
    double* X, int64_t ldx,
    double* rcond,
    double* ferr,
    double* berr )
{
    char fact_ = to_char( fact );
    char trans_ = to_char( trans );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
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
    lapack::vector< double > work( (3*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_dgtsvx(
        &fact_, &trans_, &n_, &nrhs_,
        DL,
        D,
        DU,
        DLF,
        DF,
        DUF,
        DU2,
        ipiv_ptr,
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
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gtsv
int64_t gtsvx(
    lapack::Factored fact, lapack::Op trans, int64_t n, int64_t nrhs,
    std::complex<float> const* DL,
    std::complex<float> const* D,
    std::complex<float> const* DU,
    std::complex<float>* DLF,
    std::complex<float>* DF,
    std::complex<float>* DUF,
    std::complex<float>* DU2,
    int64_t* ipiv,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float>* X, int64_t ldx,
    float* rcond,
    float* ferr,
    float* berr )
{
    char fact_ = to_char( fact );
    char trans_ = to_char( trans );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
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
    lapack::vector< float > rwork( (n) );

    LAPACK_cgtsvx(
        &fact_, &trans_, &n_, &nrhs_,
        (lapack_complex_float*) DL,
        (lapack_complex_float*) D,
        (lapack_complex_float*) DU,
        (lapack_complex_float*) DLF,
        (lapack_complex_float*) DF,
        (lapack_complex_float*) DUF,
        (lapack_complex_float*) DU2,
        ipiv_ptr,
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
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// Uses the LU factorization to compute the solution to a complex
/// system of linear equations
/// \[
///     A   X = B,
/// \]
/// \[
///     A^T X = B,
/// \]
/// or
/// \[
///     A^H X = B,
/// \]
/// where A is a tridiagonal matrix of order n and X and B are n-by-nrhs
/// matrices.
///
/// Error bounds on the solution and a condition estimate are also
/// provided.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] fact
///     Specifies whether or not the factored form of A has been
///     supplied on entry.
///     - lapack::Factored::Factored:
///         DLF, DF, DUF, DU2, and ipiv contain the factored form of A;
///         DL, D, DU, DLF, DF, DUF, DU2 and ipiv will not be modified.
///     - lapack::Factored::NotFactored:
///         The matrix will be copied to DLF, DF, and DUF and factored.
///
/// @param[in] trans
///     Specifies the form of the system of equations:
///     - lapack::Op::NoTrans:   $A   X = B$ (No transpose)
///     - lapack::Op::Trans:     $A^T X = B$ (Transpose)
///     - lapack::Op::ConjTrans: $A^H X = B$ (Conjugate transpose)
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrix B. nrhs >= 0.
///
/// @param[in] DL
///     The vector DL of length n-1.
///     The (n-1) subdiagonal elements of A.
///
/// @param[in] D
///     The vector D of length n.
///     The n diagonal elements of A.
///
/// @param[in] DU
///     The vector DU of length n-1.
///     The (n-1) superdiagonal elements of A.
///
/// @param[in,out] DLF
///     The vector DLF of length n-1.
///     - If fact = Factored, then DLF is an input argument and on entry
///     contains the (n-1) multipliers that define the matrix L from
///     the LU factorization of A as computed by `lapack::gttrf`.
///
///     - If fact = NotFactored, then DLF is an output argument and on exit
///     contains the (n-1) multipliers that define the matrix L from
///     the LU factorization of A.
///
/// @param[in,out] DF
///     The vector DF of length n.
///     - If fact = Factored, then DF is an input argument and on entry
///     contains the n diagonal elements of the upper triangular
///     matrix U from the LU factorization of A.
///
///     - If fact = NotFactored, then DF is an output argument and on exit
///     contains the n diagonal elements of the upper triangular
///     matrix U from the LU factorization of A.
///
/// @param[in,out] DUF
///     The vector DUF of length n-1.
///     - If fact = Factored, then DUF is an input argument and on entry
///     contains the (n-1) elements of the first superdiagonal of U.
///
///     - If fact = NotFactored, then DUF is an output argument and on exit
///     contains the (n-1) elements of the first superdiagonal of U.
///
/// @param[in,out] DU2
///     The vector DU2 of length n-2.
///     - If fact = Factored, then DU2 is an input argument and on entry
///     contains the (n-2) elements of the second superdiagonal of U.
///
///     - If fact = NotFactored, then DU2 is an output argument and on exit
///     contains the (n-2) elements of the second superdiagonal of U.
///
/// @param[in,out] ipiv
///     The vector ipiv of length n.
///     - If fact = Factored, then ipiv is an input argument and on entry
///     contains the pivot indices from the LU factorization of A as
///     computed by `lapack::gttrf`.
///
///     - If fact = NotFactored, then ipiv is an output argument and on exit
///     contains the pivot indices from the LU factorization of A;
///     row i of the matrix was interchanged with row ipiv(i).
///     ipiv(i) will always be either i or i+1; ipiv(i) = i indicates
///     a row interchange was not required.
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
///     The estimate of the reciprocal condition number of the matrix
///     A. If rcond is less than the machine precision (in
///     particular, if rcond = 0), the matrix is singular to working
///     precision. This condition is indicated by a return code of
///     return value > 0.
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
///     U(i,i) is exactly zero. The factorization
///     has not been completed unless i = n, but the
///     factor U is exactly singular, so the solution
///     and error bounds could not be computed.
///     rcond = 0 is returned.
/// @return = n+1: U is nonsingular, but rcond is less than machine
///     precision, meaning that the matrix is singular
///     to working precision. Nevertheless, the
///     solution and error bounds are computed because
///     there are a number of situations where the
///     computed solution can be more accurate than the
///     value of rcond would suggest.
///
/// @ingroup gtsv
int64_t gtsvx(
    lapack::Factored fact, lapack::Op trans, int64_t n, int64_t nrhs,
    std::complex<double> const* DL,
    std::complex<double> const* D,
    std::complex<double> const* DU,
    std::complex<double>* DLF,
    std::complex<double>* DF,
    std::complex<double>* DUF,
    std::complex<double>* DU2,
    int64_t* ipiv,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double>* X, int64_t ldx,
    double* rcond,
    double* ferr,
    double* berr )
{
    char fact_ = to_char( fact );
    char trans_ = to_char( trans );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
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
    lapack::vector< double > rwork( (n) );

    LAPACK_zgtsvx(
        &fact_, &trans_, &n_, &nrhs_,
        (lapack_complex_double*) DL,
        (lapack_complex_double*) D,
        (lapack_complex_double*) DU,
        (lapack_complex_double*) DLF,
        (lapack_complex_double*) DF,
        (lapack_complex_double*) DUF,
        (lapack_complex_double*) DU2,
        ipiv_ptr,
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
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

}  // namespace lapack
