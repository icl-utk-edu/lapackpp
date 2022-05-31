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
int64_t hgeqz(
    lapack::JobSchur jobschur, lapack::Job compq, lapack::Job compz, int64_t n, int64_t ilo, int64_t ihi,
    float* H, int64_t ldh,
    float* T, int64_t ldt,
    std::complex<float>* alpha,
    float* beta,
    float* Q, int64_t ldq,
    float* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ilo) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ihi) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldh) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    char jobschur_ = jobschur2char( jobschur );
    char compq_ = job_comp2char( compq );
    char compz_ = job_comp2char( compz );
    lapack_int n_ = (lapack_int) n;
    lapack_int ilo_ = (lapack_int) ilo;
    lapack_int ihi_ = (lapack_int) ihi;
    lapack_int ldh_ = (lapack_int) ldh;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldq_ = (lapack_int) ldq;
    lapack_int ldz_ = (lapack_int) ldz;
    lapack_int info_ = 0;

    // split-complex representation
    lapack::vector< float > alphar( max( 1, n ) );
    lapack::vector< float > alphai( max( 1, n ) );

    // query for workspace size
    float qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_shgeqz(
        &jobschur_, &compq_, &compz_, &n_, &ilo_, &ihi_,
        H, &ldh_,
        T, &ldt_,
        &alphar[0],
        &alphai[0],
        beta,
        Q, &ldq_,
        Z, &ldz_,
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
    lapack::vector< float > work( lwork_ );

    LAPACK_shgeqz(
        &jobschur_, &compq_, &compz_, &n_, &ilo_, &ihi_,
        H, &ldh_,
        T, &ldt_,
        &alphar[0],
        &alphai[0],
        beta,
        Q, &ldq_,
        Z, &ldz_,
        &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
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
int64_t hgeqz(
    lapack::JobSchur jobschur, lapack::Job compq, lapack::Job compz, int64_t n, int64_t ilo, int64_t ihi,
    double* H, int64_t ldh,
    double* T, int64_t ldt,
    std::complex<double>* alpha,
    double* beta,
    double* Q, int64_t ldq,
    double* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ilo) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ihi) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldh) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    char jobschur_ = jobschur2char( jobschur );
    char compq_ = job_comp2char( compq );
    char compz_ = job_comp2char( compz );
    lapack_int n_ = (lapack_int) n;
    lapack_int ilo_ = (lapack_int) ilo;
    lapack_int ihi_ = (lapack_int) ihi;
    lapack_int ldh_ = (lapack_int) ldh;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldq_ = (lapack_int) ldq;
    lapack_int ldz_ = (lapack_int) ldz;
    lapack_int info_ = 0;

    // split-complex representation
    lapack::vector< double > alphar( max( 1, n ) );
    lapack::vector< double > alphai( max( 1, n ) );

    // query for workspace size
    double qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_dhgeqz(
        &jobschur_, &compq_, &compz_, &n_, &ilo_, &ihi_,
        H, &ldh_,
        T, &ldt_,
        &alphar[0],
        &alphai[0],
        beta,
        Q, &ldq_,
        Z, &ldz_,
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
    lapack::vector< double > work( lwork_ );

    LAPACK_dhgeqz(
        &jobschur_, &compq_, &compz_, &n_, &ilo_, &ihi_,
        H, &ldh_,
        T, &ldt_,
        &alphar[0],
        &alphai[0],
        beta,
        Q, &ldq_,
        Z, &ldz_,
        &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
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
int64_t hgeqz(
    lapack::JobSchur jobschur, lapack::Job compq, lapack::Job compz, int64_t n, int64_t ilo, int64_t ihi,
    std::complex<float>* H, int64_t ldh,
    std::complex<float>* T, int64_t ldt,
    std::complex<float>* alpha,
    std::complex<float>* beta,
    std::complex<float>* Q, int64_t ldq,
    std::complex<float>* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ilo) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ihi) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldh) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    char jobschur_ = jobschur2char( jobschur );
    char compq_ = job_comp2char( compq );
    char compz_ = job_comp2char( compz );
    lapack_int n_ = (lapack_int) n;
    lapack_int ilo_ = (lapack_int) ilo;
    lapack_int ihi_ = (lapack_int) ihi;
    lapack_int ldh_ = (lapack_int) ldh;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldq_ = (lapack_int) ldq;
    lapack_int ldz_ = (lapack_int) ldz;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_chgeqz(
        &jobschur_, &compq_, &compz_, &n_, &ilo_, &ihi_,
        (lapack_complex_float*) H, &ldh_,
        (lapack_complex_float*) T, &ldt_,
        (lapack_complex_float*) alpha,
        (lapack_complex_float*) beta,
        (lapack_complex_float*) Q, &ldq_,
        (lapack_complex_float*) Z, &ldz_,
        (lapack_complex_float*) qry_work, &ineg_one,
        qry_rwork, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );
    lapack::vector< float > rwork( (n) );

    LAPACK_chgeqz(
        &jobschur_, &compq_, &compz_, &n_, &ilo_, &ihi_,
        (lapack_complex_float*) H, &ldh_,
        (lapack_complex_float*) T, &ldt_,
        (lapack_complex_float*) alpha,
        (lapack_complex_float*) beta,
        (lapack_complex_float*) Q, &ldq_,
        (lapack_complex_float*) Z, &ldz_,
        (lapack_complex_float*) &work[0], &lwork_,
        &rwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t hgeqz(
    lapack::JobSchur jobschur, lapack::Job compq, lapack::Job compz, int64_t n, int64_t ilo, int64_t ihi,
    std::complex<double>* H, int64_t ldh,
    std::complex<double>* T, int64_t ldt,
    std::complex<double>* alpha,
    std::complex<double>* beta,
    std::complex<double>* Q, int64_t ldq,
    std::complex<double>* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ilo) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ihi) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldh) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    char jobschur_ = jobschur2char( jobschur );
    char compq_ = job_comp2char( compq );
    char compz_ = job_comp2char( compz );
    lapack_int n_ = (lapack_int) n;
    lapack_int ilo_ = (lapack_int) ilo;
    lapack_int ihi_ = (lapack_int) ihi;
    lapack_int ldh_ = (lapack_int) ldh;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldq_ = (lapack_int) ldq;
    lapack_int ldz_ = (lapack_int) ldz;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_zhgeqz(
        &jobschur_, &compq_, &compz_, &n_, &ilo_, &ihi_,
        (lapack_complex_double*) H, &ldh_,
        (lapack_complex_double*) T, &ldt_,
        (lapack_complex_double*) alpha,
        (lapack_complex_double*) beta,
        (lapack_complex_double*) Q, &ldq_,
        (lapack_complex_double*) Z, &ldz_,
        (lapack_complex_double*) qry_work, &ineg_one,
        qry_rwork, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );
    lapack::vector< double > rwork( (n) );

    LAPACK_zhgeqz(
        &jobschur_, &compq_, &compz_, &n_, &ilo_, &ihi_,
        (lapack_complex_double*) H, &ldh_,
        (lapack_complex_double*) T, &ldt_,
        (lapack_complex_double*) alpha,
        (lapack_complex_double*) beta,
        (lapack_complex_double*) Q, &ldq_,
        (lapack_complex_double*) Z, &ldz_,
        (lapack_complex_double*) &work[0], &lwork_,
        &rwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
