// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#if LAPACK_VERSION >= 30500  // >= 3.5

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t orcsd2by1(
    lapack::Job jobu1, lapack::Job jobu2, lapack::Job jobv1t, int64_t m, int64_t p, int64_t q,
    float* X11, int64_t ldx11,
    float* X21, int64_t ldx21,
    float* theta,
    float* U1, int64_t ldu1,
    float* U2, int64_t ldu2,
    float* V1T, int64_t ldv1t )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(q) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldx11) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldx21) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu1) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu2) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv1t) > std::numeric_limits<lapack_int>::max() );
    }
    char jobu1_ = job_csd2char( jobu1 );
    char jobu2_ = job_csd2char( jobu2 );
    char jobv1t_ = job_csd2char( jobv1t );
    lapack_int m_ = (lapack_int) m;
    lapack_int p_ = (lapack_int) p;
    lapack_int q_ = (lapack_int) q;
    lapack_int ldx11_ = (lapack_int) ldx11;
    lapack_int ldx21_ = (lapack_int) ldx21;
    lapack_int ldu1_ = (lapack_int) ldu1;
    lapack_int ldu2_ = (lapack_int) ldu2;
    lapack_int ldv1t_ = (lapack_int) ldv1t;
    lapack_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_sorcsd2by1(
        &jobu1_, &jobu2_, &jobv1t_, &m_, &p_, &q_,
        X11, &ldx11_,
        X21, &ldx21_,
        theta,
        U1, &ldu1_,
        U2, &ldu2_,
        V1T, &ldv1t_,
        qry_work, &ineg_one,
        qry_iwork, &info_
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
    lapack::vector< lapack_int > iwork( (m - min( p, min( m-p, min( q, m-q )))) );

    LAPACK_sorcsd2by1(
        &jobu1_, &jobu2_, &jobv1t_, &m_, &p_, &q_,
        X11, &ldx11_,
        X21, &ldx21_,
        theta,
        U1, &ldu1_,
        U2, &ldu2_,
        V1T, &ldv1t_,
        &work[0], &lwork_,
        &iwork[0], &info_
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
int64_t orcsd2by1(
    lapack::Job jobu1, lapack::Job jobu2, lapack::Job jobv1t, int64_t m, int64_t p, int64_t q,
    double* X11, int64_t ldx11,
    double* X21, int64_t ldx21,
    double* theta,
    double* U1, int64_t ldu1,
    double* U2, int64_t ldu2,
    double* V1T, int64_t ldv1t )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(q) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldx11) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldx21) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu1) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu2) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv1t) > std::numeric_limits<lapack_int>::max() );
    }
    char jobu1_ = job_csd2char( jobu1 );
    char jobu2_ = job_csd2char( jobu2 );
    char jobv1t_ = job_csd2char( jobv1t );
    lapack_int m_ = (lapack_int) m;
    lapack_int p_ = (lapack_int) p;
    lapack_int q_ = (lapack_int) q;
    lapack_int ldx11_ = (lapack_int) ldx11;
    lapack_int ldx21_ = (lapack_int) ldx21;
    lapack_int ldu1_ = (lapack_int) ldu1;
    lapack_int ldu2_ = (lapack_int) ldu2;
    lapack_int ldv1t_ = (lapack_int) ldv1t;
    lapack_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_dorcsd2by1(
        &jobu1_, &jobu2_, &jobv1t_, &m_, &p_, &q_,
        X11, &ldx11_,
        X21, &ldx21_,
        theta,
        U1, &ldu1_,
        U2, &ldu2_,
        V1T, &ldv1t_,
        qry_work, &ineg_one,
        qry_iwork, &info_
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
    lapack::vector< lapack_int > iwork( (m - min( p, min( m-p, min( q, m-q )))) );

    LAPACK_dorcsd2by1(
        &jobu1_, &jobu2_, &jobv1t_, &m_, &p_, &q_,
        X11, &ldx11_,
        X21, &ldx21_,
        theta,
        U1, &ldu1_,
        U2, &ldu2_,
        V1T, &ldv1t_,
        &work[0], &lwork_,
        &iwork[0], &info_
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

#endif  // LAPACK >= 3.5.0
