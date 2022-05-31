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
int64_t gees(
    lapack::Job jobvs, lapack::Sort sort, lapack_s_select2 select, int64_t n,
    float* A, int64_t lda,
    int64_t* sdim,
    std::complex<float>* W,
    float* VS, int64_t ldvs )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvs) > std::numeric_limits<lapack_int>::max() );
    }
    char jobvs_ = job2char( jobvs );
    char sort_ = sort2char( sort );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int sdim_ = (lapack_int) *sdim;
    lapack_int ldvs_ = (lapack_int) ldvs;
    lapack_int info_ = 0;

    // split-complex representation
    lapack::vector< float > WR( max( 1, n ) );
    lapack::vector< float > WI( max( 1, n ) );

    // query for workspace size
    float qry_work[1];
    lapack_int qry_bwork[1];
    lapack_int ineg_one = -1;
    LAPACK_sgees(
        &jobvs_, &sort_,
        select, &n_,
        A, &lda_, &sdim_,
        &WR[0],
        &WI[0],
        VS, &ldvs_,
        qry_work, &ineg_one,
        qry_bwork, &info_
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
    lapack::vector< lapack_int > bwork( (n) );

    LAPACK_sgees(
        &jobvs_, &sort_,
        select, &n_,
        A, &lda_, &sdim_,
        &WR[0],
        &WI[0],
        VS, &ldvs_,
        &work[0], &lwork_,
        &bwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    // merge split-complex representation
    for (int64_t i = 0; i < n; ++i) {
        W[i] = std::complex<float>( WR[i], WI[i] );
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gees(
    lapack::Job jobvs, lapack::Sort sort, lapack_d_select2 select, int64_t n,
    double* A, int64_t lda,
    int64_t* sdim,
    std::complex<double>* W,
    double* VS, int64_t ldvs )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvs) > std::numeric_limits<lapack_int>::max() );
    }
    char jobvs_ = job2char( jobvs );
    char sort_ = sort2char( sort );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int sdim_ = (lapack_int) *sdim;
    lapack_int ldvs_ = (lapack_int) ldvs;
    lapack_int info_ = 0;

    // split-complex representation
    lapack::vector< double > WR( max( 1, n ) );
    lapack::vector< double > WI( max( 1, n ) );

    // query for workspace size
    double qry_work[1];
    lapack_int qry_bwork[1];
    lapack_int ineg_one = -1;
    LAPACK_dgees(
        &jobvs_, &sort_,
        select, &n_,
        A, &lda_, &sdim_,
        &WR[0],
        &WI[0],
        VS, &ldvs_,
        qry_work, &ineg_one,
        qry_bwork, &info_
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
    lapack::vector< lapack_int > bwork( (n) );

    LAPACK_dgees(
        &jobvs_, &sort_,
        select, &n_,
        A, &lda_, &sdim_,
        &WR[0],
        &WI[0],
        VS, &ldvs_,
        &work[0], &lwork_,
        &bwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    // merge split-complex representation
    for (int64_t i = 0; i < n; ++i) {
        W[i] = std::complex<double>( WR[i], WI[i] );
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gees(
    lapack::Job jobvs, lapack::Sort sort, lapack_c_select1 select, int64_t n,
    std::complex<float>* A, int64_t lda,
    int64_t* sdim,
    std::complex<float>* W,
    std::complex<float>* VS, int64_t ldvs )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvs) > std::numeric_limits<lapack_int>::max() );
    }
    char jobvs_ = job2char( jobvs );
    char sort_ = sort2char( sort );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int sdim_ = (lapack_int) *sdim;
    lapack_int ldvs_ = (lapack_int) ldvs;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    lapack_int qry_bwork[1];
    lapack_int ineg_one = -1;
    LAPACK_cgees(
        &jobvs_, &sort_,
        (LAPACK_C_SELECT1) select, &n_,
        (lapack_complex_float*) A, &lda_, &sdim_,
        (lapack_complex_float*) W,
        (lapack_complex_float*) VS, &ldvs_,
        (lapack_complex_float*) qry_work, &ineg_one,
        qry_rwork,
        qry_bwork, &info_
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
    lapack::vector< float > rwork( (n) );
    lapack::vector< lapack_int > bwork( (n) );

    LAPACK_cgees(
        &jobvs_, &sort_,
        (LAPACK_C_SELECT1) select, &n_,
        (lapack_complex_float*) A, &lda_, &sdim_,
        (lapack_complex_float*) W,
        (lapack_complex_float*) VS, &ldvs_,
        (lapack_complex_float*) &work[0], &lwork_,
        &rwork[0],
        &bwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gees(
    lapack::Job jobvs, lapack::Sort sort, lapack_z_select1 select, int64_t n,
    std::complex<double>* A, int64_t lda,
    int64_t* sdim,
    std::complex<double>* W,
    std::complex<double>* VS, int64_t ldvs )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvs) > std::numeric_limits<lapack_int>::max() );
    }
    char jobvs_ = job2char( jobvs );
    char sort_ = sort2char( sort );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int sdim_ = (lapack_int) *sdim;
    lapack_int ldvs_ = (lapack_int) ldvs;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    lapack_int qry_bwork[1];
    lapack_int ineg_one = -1;
    LAPACK_zgees(
        &jobvs_, &sort_,
        (LAPACK_Z_SELECT1) select, &n_,
        (lapack_complex_double*) A, &lda_, &sdim_,
        (lapack_complex_double*) W,
        (lapack_complex_double*) VS, &ldvs_,
        (lapack_complex_double*) qry_work, &ineg_one,
        qry_rwork,
        qry_bwork, &info_
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
    lapack::vector< double > rwork( (n) );
    lapack::vector< lapack_int > bwork( (n) );

    LAPACK_zgees(
        &jobvs_, &sort_,
        (LAPACK_Z_SELECT1) select, &n_,
        (lapack_complex_double*) A, &lda_, &sdim_,
        (lapack_complex_double*) W,
        (lapack_complex_double*) VS, &ldvs_,
        (lapack_complex_double*) &work[0], &lwork_,
        &rwork[0],
        &bwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

}  // namespace lapack
