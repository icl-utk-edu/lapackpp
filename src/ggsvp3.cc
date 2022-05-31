// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#if LAPACK_VERSION >= 30600  // >= v3.6

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t ggsvp3(
    lapack::Job jobu, lapack::Job jobv, lapack::Job jobq, int64_t m, int64_t p, int64_t n,
    float* A, int64_t lda,
    float* B, int64_t ldb, float tola, float tolb,
    int64_t* k,
    int64_t* l,
    float* U, int64_t ldu,
    float* V, int64_t ldv,
    float* Q, int64_t ldq,
    float* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<lapack_int>::max() );
    }
    char jobu_ = jobu2char( jobu );
    char jobv_ = job2char( jobv );
    char jobq_ = jobq2char( jobq );
    lapack_int m_ = (lapack_int) m;
    lapack_int p_ = (lapack_int) p;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int k_ = (lapack_int) *k;
    lapack_int l_ = (lapack_int) *l;
    lapack_int ldu_ = (lapack_int) ldu;
    lapack_int ldv_ = (lapack_int) ldv;
    lapack_int ldq_ = (lapack_int) ldq;
    lapack_int info_ = 0;

    // query for workspace size
    lapack_int qry_iwork[1];
    float qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_sggsvp3(
        &jobu_, &jobv_, &jobq_, &m_, &p_, &n_,
        A, &lda_,
        B, &ldb_, &tola, &tolb, &k_, &l_,
        U, &ldu_,
        V, &ldv_,
        Q, &ldq_,
        qry_iwork,
        tau,
        qry_work, &ineg_one, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< lapack_int > iwork( (n) );
    lapack::vector< float > work( lwork_ );

    LAPACK_sggsvp3(
        &jobu_, &jobv_, &jobq_, &m_, &p_, &n_,
        A, &lda_,
        B, &ldb_, &tola, &tolb, &k_, &l_,
        U, &ldu_,
        V, &ldv_,
        Q, &ldq_,
        &iwork[0],
        tau,
        &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    *k = k_;
    *l = l_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggsvp3(
    lapack::Job jobu, lapack::Job jobv, lapack::Job jobq, int64_t m, int64_t p, int64_t n,
    double* A, int64_t lda,
    double* B, int64_t ldb, double tola, double tolb,
    int64_t* k,
    int64_t* l,
    double* U, int64_t ldu,
    double* V, int64_t ldv,
    double* Q, int64_t ldq,
    double* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<lapack_int>::max() );
    }
    char jobu_ = jobu2char( jobu );
    char jobv_ = job2char( jobv );
    char jobq_ = jobq2char( jobq );
    lapack_int m_ = (lapack_int) m;
    lapack_int p_ = (lapack_int) p;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int k_ = (lapack_int) *k;
    lapack_int l_ = (lapack_int) *l;
    lapack_int ldu_ = (lapack_int) ldu;
    lapack_int ldv_ = (lapack_int) ldv;
    lapack_int ldq_ = (lapack_int) ldq;
    lapack_int info_ = 0;

    // query for workspace size
    lapack_int qry_iwork[1];
    double qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_dggsvp3(
        &jobu_, &jobv_, &jobq_, &m_, &p_, &n_,
        A, &lda_,
        B, &ldb_, &tola, &tolb, &k_, &l_,
        U, &ldu_,
        V, &ldv_,
        Q, &ldq_,
        qry_iwork,
        tau,
        qry_work, &ineg_one, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< lapack_int > iwork( (n) );
    lapack::vector< double > work( lwork_ );

    LAPACK_dggsvp3(
        &jobu_, &jobv_, &jobq_, &m_, &p_, &n_,
        A, &lda_,
        B, &ldb_, &tola, &tolb, &k_, &l_,
        U, &ldu_,
        V, &ldv_,
        Q, &ldq_,
        &iwork[0],
        tau,
        &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    *k = k_;
    *l = l_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggsvp3(
    lapack::Job jobu, lapack::Job jobv, lapack::Job jobq, int64_t m, int64_t p, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb, float tola, float tolb,
    int64_t* k,
    int64_t* l,
    std::complex<float>* U, int64_t ldu,
    std::complex<float>* V, int64_t ldv,
    std::complex<float>* Q, int64_t ldq,
    std::complex<float>* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<lapack_int>::max() );
    }
    char jobu_ = jobu2char( jobu );
    char jobv_ = job2char( jobv );
    char jobq_ = jobq2char( jobq );
    lapack_int m_ = (lapack_int) m;
    lapack_int p_ = (lapack_int) p;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int k_ = (lapack_int) *k;
    lapack_int l_ = (lapack_int) *l;
    lapack_int ldu_ = (lapack_int) ldu;
    lapack_int ldv_ = (lapack_int) ldv;
    lapack_int ldq_ = (lapack_int) ldq;
    lapack_int info_ = 0;

    // query for workspace size
    lapack_int qry_iwork[1];
    float qry_rwork[1];
    std::complex<float> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_cggsvp3(
        &jobu_, &jobv_, &jobq_, &m_, &p_, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_, &tola, &tolb, &k_, &l_,
        (lapack_complex_float*) U, &ldu_,
        (lapack_complex_float*) V, &ldv_,
        (lapack_complex_float*) Q, &ldq_,
        qry_iwork,
        qry_rwork,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) qry_work, &ineg_one, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< lapack_int > iwork( (n) );
    lapack::vector< float > rwork( (2*n) );
    lapack::vector< std::complex<float> > work( lwork_ );

    LAPACK_cggsvp3(
        &jobu_, &jobv_, &jobq_, &m_, &p_, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_, &tola, &tolb, &k_, &l_,
        (lapack_complex_float*) U, &ldu_,
        (lapack_complex_float*) V, &ldv_,
        (lapack_complex_float*) Q, &ldq_,
        &iwork[0],
        &rwork[0],
        (lapack_complex_float*) tau,
        (lapack_complex_float*) &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    *k = k_;
    *l = l_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggsvp3(
    lapack::Job jobu, lapack::Job jobv, lapack::Job jobq, int64_t m, int64_t p, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb, double tola, double tolb,
    int64_t* k,
    int64_t* l,
    std::complex<double>* U, int64_t ldu,
    std::complex<double>* V, int64_t ldv,
    std::complex<double>* Q, int64_t ldq,
    std::complex<double>* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<lapack_int>::max() );
    }
    char jobu_ = jobu2char( jobu );
    char jobv_ = job2char( jobv );
    char jobq_ = jobq2char( jobq );
    lapack_int m_ = (lapack_int) m;
    lapack_int p_ = (lapack_int) p;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int k_ = (lapack_int) *k;
    lapack_int l_ = (lapack_int) *l;
    lapack_int ldu_ = (lapack_int) ldu;
    lapack_int ldv_ = (lapack_int) ldv;
    lapack_int ldq_ = (lapack_int) ldq;
    lapack_int info_ = 0;

    // query for workspace size
    lapack_int qry_iwork[1];
    double qry_rwork[1];
    std::complex<double> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_zggsvp3(
        &jobu_, &jobv_, &jobq_, &m_, &p_, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_, &tola, &tolb, &k_, &l_,
        (lapack_complex_double*) U, &ldu_,
        (lapack_complex_double*) V, &ldv_,
        (lapack_complex_double*) Q, &ldq_,
        qry_iwork,
        qry_rwork,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) qry_work, &ineg_one, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< lapack_int > iwork( (n) );
    lapack::vector< double > rwork( (2*n) );
    lapack::vector< std::complex<double> > work( lwork_ );

    LAPACK_zggsvp3(
        &jobu_, &jobv_, &jobq_, &m_, &p_, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_, &tola, &tolb, &k_, &l_,
        (lapack_complex_double*) U, &ldu_,
        (lapack_complex_double*) V, &ldv_,
        (lapack_complex_double*) Q, &ldq_,
        &iwork[0],
        &rwork[0],
        (lapack_complex_double*) tau,
        (lapack_complex_double*) &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    *k = k_;
    *l = l_;
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= v3.6
