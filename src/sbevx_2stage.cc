// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#if LAPACK_VERSION >= 30700  // >= 3.7

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t sbevx_2stage(
    lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n, int64_t kd,
    float* AB, int64_t ldab,
    float* Q, int64_t ldq, float vl, float vu, int64_t il, int64_t iu, float abstol,
    int64_t* nfound,
    float* W,
    float* Z, int64_t ldz,
    int64_t* ifail )
{
    char jobz_ = to_char( jobz );
    char range_ = to_char( range );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int il_ = to_lapack_int( il );
    lapack_int iu_ = to_lapack_int( iu );
    lapack_int nfound_ = 0;
    lapack_int ldz_ = to_lapack_int( ldz );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ifail_( (n) );
        lapack_int* ifail_ptr = &ifail_[0];
    #else
        lapack_int* ifail_ptr = ifail;
    #endif
    lapack_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_ssbevx_2stage(
        &jobz_, &range_, &uplo_, &n_, &kd_,
        AB, &ldab_,
        Q, &ldq_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        Z, &ldz_,
        qry_work, &ineg_one,
        qry_iwork,
        ifail_ptr, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< float > work( lwork_ );
    lapack::vector< lapack_int > iwork( (5*n) );

    LAPACK_ssbevx_2stage(
        &jobz_, &range_, &uplo_, &n_, &kd_,
        AB, &ldab_,
        Q, &ldq_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        Z, &ldz_,
        &work[0], &lwork_,
        &iwork[0],
        ifail_ptr, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    *nfound = nfound_;
    #ifndef LAPACK_ILP64
        if (jobz != Job::NoVec) {
            std::copy( &ifail_[ 0 ], &ifail_[ nfound_ ], ifail );
        }
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t sbevx_2stage(
    lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n, int64_t kd,
    double* AB, int64_t ldab,
    double* Q, int64_t ldq, double vl, double vu, int64_t il, int64_t iu, double abstol,
    int64_t* nfound,
    double* W,
    double* Z, int64_t ldz,
    int64_t* ifail )
{
    char jobz_ = to_char( jobz );
    char range_ = to_char( range );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int il_ = to_lapack_int( il );
    lapack_int iu_ = to_lapack_int( iu );
    lapack_int nfound_ = 0;
    lapack_int ldz_ = to_lapack_int( ldz );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ifail_( (n) );
        lapack_int* ifail_ptr = &ifail_[0];
    #else
        lapack_int* ifail_ptr = ifail;
    #endif
    lapack_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_dsbevx_2stage(
        &jobz_, &range_, &uplo_, &n_, &kd_,
        AB, &ldab_,
        Q, &ldq_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        Z, &ldz_,
        qry_work, &ineg_one,
        qry_iwork,
        ifail_ptr, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< double > work( lwork_ );
    lapack::vector< lapack_int > iwork( (5*n) );

    LAPACK_dsbevx_2stage(
        &jobz_, &range_, &uplo_, &n_, &kd_,
        AB, &ldab_,
        Q, &ldq_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        Z, &ldz_,
        &work[0], &lwork_,
        &iwork[0],
        ifail_ptr, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    *nfound = nfound_;
    #ifndef LAPACK_ILP64
        if (jobz != Job::NoVec) {
            std::copy( &ifail_[ 0 ], &ifail_[ nfound_ ], ifail );
        }
    #endif
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.7
