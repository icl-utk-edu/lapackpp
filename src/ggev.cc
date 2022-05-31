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
int64_t ggev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    std::complex<float>* alpha,
    float* beta,
    float* VL, int64_t ldvl,
    float* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvl) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvr) > std::numeric_limits<lapack_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int ldvl_ = (lapack_int) ldvl;
    lapack_int ldvr_ = (lapack_int) ldvr;
    lapack_int info_ = 0;

    // split-complex representation
    lapack::vector< float > alphar( max( 1, n ) );
    lapack::vector< float > alphai( max( 1, n ) );

    // query for workspace size
    float qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_sggev(
        &jobvl_, &jobvr_, &n_,
        A, &lda_,
        B, &ldb_,
        &alphar[0], &alphai[0],
        beta,
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

    LAPACK_sggev(
        &jobvl_, &jobvr_, &n_,
        A, &lda_,
        B, &ldb_,
        &alphar[0], &alphai[0],
        beta,
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
        alpha[i] = std::complex<float>( alphar[i], alphai[i] );
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    std::complex<double>* alpha,
    double* beta,
    double* VL, int64_t ldvl,
    double* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvl) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvr) > std::numeric_limits<lapack_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int ldvl_ = (lapack_int) ldvl;
    lapack_int ldvr_ = (lapack_int) ldvr;
    lapack_int info_ = 0;

    // split-complex representation
    lapack::vector< double > alphar( max( 1, n ) );
    lapack::vector< double > alphai( max( 1, n ) );

    // query for workspace size
    double qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_dggev(
        &jobvl_, &jobvr_, &n_,
        A, &lda_,
        B, &ldb_,
        &alphar[0], &alphai[0],
        beta,
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

    LAPACK_dggev(
        &jobvl_, &jobvr_, &n_,
        A, &lda_,
        B, &ldb_,
        &alphar[0], &alphai[0],
        beta,
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
        alpha[i] = std::complex<double>( alphar[i], alphai[i] );
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* alpha,
    std::complex<float>* beta,
    std::complex<float>* VL, int64_t ldvl,
    std::complex<float>* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvl) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvr) > std::numeric_limits<lapack_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int ldvl_ = (lapack_int) ldvl;
    lapack_int ldvr_ = (lapack_int) ldvr;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_cggev(
        &jobvl_, &jobvr_, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) alpha,
        (lapack_complex_float*) beta,
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
    lapack::vector< float > rwork( (8*n) );

    LAPACK_cggev(
        &jobvl_, &jobvr_, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) alpha,
        (lapack_complex_float*) beta,
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
int64_t ggev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* alpha,
    std::complex<double>* beta,
    std::complex<double>* VL, int64_t ldvl,
    std::complex<double>* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvl) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvr) > std::numeric_limits<lapack_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int ldvl_ = (lapack_int) ldvl;
    lapack_int ldvr_ = (lapack_int) ldvr;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_zggev(
        &jobvl_, &jobvr_, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) alpha,
        (lapack_complex_double*) beta,
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
    lapack::vector< double > rwork( (8*n) );

    LAPACK_zggev(
        &jobvl_, &jobvr_, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) alpha,
        (lapack_complex_double*) beta,
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
