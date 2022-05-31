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
/// @ingroup geev
int64_t geev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    float* A, int64_t lda,
    std::complex<float>* W,
    float* VL, int64_t ldvl,
    float* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvl) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvr) > std::numeric_limits<lapack_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldvl_ = (lapack_int) ldvl;
    lapack_int ldvr_ = (lapack_int) ldvr;
    lapack_int info_ = 0;

    // split-complex representation
    lapack::vector< float > WR( max( 1, n ) );
    lapack::vector< float > WI( max( 1, n ) );

    // query for workspace size
    float qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_sgeev(
        &jobvl_, &jobvr_, &n_,
        A, &lda_,
        &WR[0], &WI[0],
        VL, &ldvl_,
        VR, &ldvr_,
        qry_work, &ineg_one, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< float > work( lwork_ );

    LAPACK_sgeev(
        &jobvl_, &jobvr_, &n_,
        A, &lda_,
        &WR[0], &WI[0],
        VL, &ldvl_,
        VR, &ldvr_,
        &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    // merge split-complex representation
    for (int64_t i = 0; i < n; ++i) {
        W[i] = std::complex<float>( WR[i], WI[i] );
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geev
int64_t geev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    double* A, int64_t lda,
    std::complex<double>* W,
    double* VL, int64_t ldvl,
    double* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvl) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvr) > std::numeric_limits<lapack_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldvl_ = (lapack_int) ldvl;
    lapack_int ldvr_ = (lapack_int) ldvr;
    lapack_int info_ = 0;

    // split-complex representation
    lapack::vector< double > WR( max( 1, n ) );
    lapack::vector< double > WI( max( 1, n ) );

    // query for workspace size
    double qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_dgeev(
        &jobvl_, &jobvr_, &n_,
        A, &lda_,
        &WR[0], &WI[0],
        VL, &ldvl_,
        VR, &ldvr_,
        qry_work, &ineg_one, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< double > work( lwork_ );

    LAPACK_dgeev(
        &jobvl_, &jobvr_, &n_,
        A, &lda_,
        &WR[0], &WI[0],
        VL, &ldvl_,
        VR, &ldvr_,
        &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    // merge split-complex representation
    for (int64_t i = 0; i < n; ++i) {
        W[i] = std::complex<double>( WR[i], WI[i] );
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geev
int64_t geev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* W,
    std::complex<float>* VL, int64_t ldvl,
    std::complex<float>* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvl) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvr) > std::numeric_limits<lapack_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldvl_ = (lapack_int) ldvl;
    lapack_int ldvr_ = (lapack_int) ldvr;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_cgeev(
        &jobvl_, &jobvr_, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) W,
        (lapack_complex_float*) VL, &ldvl_,
        (lapack_complex_float*) VR, &ldvr_,
        (lapack_complex_float*) qry_work, &ineg_one,
        qry_rwork, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );
    lapack::vector< float > rwork( (2*n) );

    LAPACK_cgeev(
        &jobvl_, &jobvr_, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) W,
        (lapack_complex_float*) VL, &ldvl_,
        (lapack_complex_float*) VR, &ldvr_,
        (lapack_complex_float*) &work[0], &lwork_,
        &rwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes for an n-by-n nonsymmetric matrix A, the
/// eigenvalues and, optionally, the left and/or right eigenvectors.
///
/// The right eigenvector v_j of A satisfies
/// \[
///     A v_j = \lambda_j v_j
/// \]
/// where $\lambda_j$ is its eigenvalue.
/// The left eigenvector $u_j$ of A satisfies
/// \[
///     u_j^H A = \lambda_j u_j^H
/// \]
/// where $u_j^H$ denotes the conjugate transpose of $u_j$.
///
/// The computed eigenvectors are normalized to have Euclidean norm
/// equal to 1 and largest component real.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] jobvl
///     - lapack::Job::NoVec: left eigenvectors of A are not computed;
///     - lapack::Job::Vec:   left eigenvectors of are computed.
///
/// @param[in] jobvr
///     - lapack::Job::NoVec: right eigenvectors of A are not computed;
///     - lapack::Job::Vec:   right eigenvectors of A are computed.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     On entry, the n-by-n matrix A.
///     On exit, A has been overwritten.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[out] W
///     The vector W of length n.
///     W contains the computed eigenvalues.
///     \n
///     Note: In LAPACK++, W is always complex, whereas LAPACK with a
///     real matrix A uses a split-complex representation (WR, WI) for W.
///
/// @param[out] VL
///     The n-by-n matrix VL, stored in an ldvl-by-n array.
///     - If jobvl = Vec, the left eigenvectors $u_j$ are stored one
///     after another in the columns of VL, in the same order
///     as their eigenvalues.
///
///     - If jobvl = NoVec, VL is not referenced.
///
///     - For std::complex versions:
///     $u_j$ = VL(:,j), the j-th column of VL.
///
///     - For real (float, double) versions:
///       + If the j-th eigenvalue is real, then
///         $u_j$ = VL(:,j),
///         the j-th column of VL.
///       + If the j-th and (j+1)-st eigenvalues form a complex
///         conjugate pair, then
///         $u_j    $ = VL(:,j) + i*VL(:,j+1) and
///         $u_{j+1}$ = VL(:,j) - i*VL(:,j+1).
///
/// @param[in] ldvl
///     The leading dimension of the array VL. ldvl >= 1;
///     if jobvl = Vec, ldvl >= n.
///
/// @param[out] VR
///     The n-by-n matrix VR, stored in an ldvr-by-n array.
///     - If jobvr = Vec, the right eigenvectors $v_j$ are stored one
///     after another in the columns of VR, in the same order
///     as their eigenvalues.
///
///     - If jobvr = NoVec, VR is not referenced.
///
///     - For std::complex versions:
///     $v_j$ = VR(:,j), the j-th column of VR.
///
///     - For real (float, double) versions:
///       + If the j-th eigenvalue is real, then
///         $v_j$ = VR(:,j),
///         the j-th column of VR.
///       + If the j-th and (j+1)-st eigenvalues form a complex
///         conjugate pair, then
///         $v_j    $ = VR(:,j) + i*VR(:,j+1) and
///         $v_{j+1}$ = VR(:,j) - i*VR(:,j+1).
///
/// @param[in] ldvr
///     The leading dimension of the array VR. ldvr >= 1;
///     if jobvr = Vec, ldvr >= n.
///
/// @return = 0: successful exit
/// @return > 0: if return value = i, the QR algorithm failed to compute all the
///              eigenvalues, and no eigenvectors have been computed;
///              elements i+1:n of W contain eigenvalues which have
///              converged.
///
/// @ingroup geev
int64_t geev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* W,
    std::complex<double>* VL, int64_t ldvl,
    std::complex<double>* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvl) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvr) > std::numeric_limits<lapack_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldvl_ = (lapack_int) ldvl;
    lapack_int ldvr_ = (lapack_int) ldvr;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_zgeev(
        &jobvl_, &jobvr_, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) W,
        (lapack_complex_double*) VL, &ldvl_,
        (lapack_complex_double*) VR, &ldvr_,
        (lapack_complex_double*) qry_work, &ineg_one,
        qry_rwork, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );
    lapack::vector< double > rwork( (2*n) );

    LAPACK_zgeev(
        &jobvl_, &jobvr_, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) W,
        (lapack_complex_double*) VL, &ldvl_,
        (lapack_complex_double*) VR, &ldvr_,
        (lapack_complex_double*) &work[0], &lwork_,
        &rwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
