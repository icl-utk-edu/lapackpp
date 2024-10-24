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
/// @ingroup hesv
int64_t hesvx(
    lapack::Factored fact, lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<float> const* A, int64_t lda,
    std::complex<float>* AF, int64_t ldaf,
    int64_t* ipiv,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float>* X, int64_t ldx,
    float* rcond,
    float* ferr,
    float* berr )
{
    char fact_ = to_char( fact );
    char uplo_ = to_char( uplo );
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

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_chesvx(
        &fact_, &uplo_, &n_, &nrhs_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) AF, &ldaf_,
        ipiv_ptr,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) X, &ldx_, rcond,
        ferr,
        berr,
        (lapack_complex_float*) qry_work, &ineg_one,
        qry_rwork, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );
    lapack::vector< float > rwork( (n) );

    LAPACK_chesvx(
        &fact_, &uplo_, &n_, &nrhs_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) AF, &ldaf_,
        ipiv_ptr,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) X, &ldx_, rcond,
        ferr,
        berr,
        (lapack_complex_float*) &work[0], &lwork_,
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
/// Uses the diagonal pivoting factorization to compute the
/// solution to a system of linear equations
///     $A X = B$,
/// where A is an n-by-n Hermitian matrix and X and B are n-by-nrhs
/// matrices.
///
/// Error bounds on the solution and a condition estimate are also
/// provided.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this is an alias for `lapack::sysvx`.
///
/// @param[in] fact
///     Whether or not the factored form of A has been supplied on entry.
///     - lapack::Factored::Factored:
///             On entry, AF and ipiv contain the factored form of A.
///             A, AF and ipiv will not be modified.
///     - lapack::Factored::NotFactored:
///             The matrix A will be copied to AF and factored.
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
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     The Hermitian matrix A.
///     - If uplo = Upper, the leading n-by-n
///     upper triangular part of A contains the upper triangular part
///     of the matrix A, and the strictly lower triangular part of A
///     is not referenced.
///
///     - If uplo = Lower, the leading n-by-n lower
///     triangular part of A contains the lower triangular part of
///     the matrix A, and the strictly upper triangular part of A is
///     not referenced.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[in,out] AF
///     The n-by-n matrix AF, stored in an ldaf-by-n array.
///     - If fact = Factored, then AF is an input argument and on entry
///     contains the block diagonal matrix D and the multipliers used
///     to obtain the factor U or L from the factorization
///     $A = U D U^H$ or $A = L D L^H$ as computed by `lapack::hetrf`.
///
///     - If fact = NotFactored, then AF is an output argument and on exit
///     returns the block diagonal matrix D and the multipliers used
///     to obtain the factor U or L from the factorization
///     $A = U D U^H$ or $A = L D L^H$.
///
/// @param[in] ldaf
///     The leading dimension of the array AF. ldaf >= max(1,n).
///
/// @param[in,out] ipiv
///     The vector ipiv of length n.
///     - If fact = Factored, then ipiv is an input argument and on entry
///     contains details of the interchanges and the block structure
///     of D, as determined by `lapack::hetrf`.
///     If ipiv(k) > 0, then rows and columns k and ipiv(k) were
///     interchanged and D(k,k) is a 1-by-1 diagonal block.
///
///     - If uplo = Upper and ipiv(k) = ipiv(k-1) < 0, then rows and
///     columns k-1 and -ipiv(k) were interchanged and
///     D(k-1:k,k-1:k) is a 2-by-2 diagonal block.
///
///     - If uplo = Lower and ipiv(k) = ipiv(k+1) < 0, then rows and
///     columns k+1 and -ipiv(k) were interchanged and
///     D(k:k+1,k:k+1) is a 2-by-2 diagonal block.
///
///     - If fact = NotFactored, then ipiv is an output argument and on exit
///     contains details of the interchanges and the block structure
///     of D, as determined by `lapack::hetrf`.
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
///     precision. This condition is indicated by a return value > 0.
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
/// @return > 0 and <= n: if return value = i, D(i,i) is exactly zero. The factorization
///             has been completed but the factor D is exactly
///             singular, so the solution and error bounds could
///             not be computed. rcond = 0 is returned.
/// @return = n+1: D is nonsingular, but rcond is less than machine
///             precision, meaning that the matrix is singular
///             to working precision. Nevertheless, the
///             solution and error bounds are computed because
///             there are a number of situations where the
///             computed solution can be more accurate than the
///             value of rcond would suggest.
///
/// @ingroup hesv
int64_t hesvx(
    lapack::Factored fact, lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<double> const* A, int64_t lda,
    std::complex<double>* AF, int64_t ldaf,
    int64_t* ipiv,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double>* X, int64_t ldx,
    double* rcond,
    double* ferr,
    double* berr )
{
    char fact_ = to_char( fact );
    char uplo_ = to_char( uplo );
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

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_zhesvx(
        &fact_, &uplo_, &n_, &nrhs_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) AF, &ldaf_,
        ipiv_ptr,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) X, &ldx_, rcond,
        ferr,
        berr,
        (lapack_complex_double*) qry_work, &ineg_one,
        qry_rwork, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );
    lapack::vector< double > rwork( (n) );

    LAPACK_zhesvx(
        &fact_, &uplo_, &n_, &nrhs_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) AF, &ldaf_,
        ipiv_ptr,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) X, &ldx_, rcond,
        ferr,
        berr,
        (lapack_complex_double*) &work[0], &lwork_,
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
