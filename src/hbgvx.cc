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
int64_t hbgvx(
    lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n, int64_t ka, int64_t kb,
    std::complex<float>* AB, int64_t ldab,
    std::complex<float>* BB, int64_t ldbb,
    std::complex<float>* Q, int64_t ldq, float vl, float vu, int64_t il, int64_t iu, float abstol,
    int64_t* nfound,
    float* W,
    std::complex<float>* Z, int64_t ldz,
    int64_t* ifail )
{
    char jobz_ = to_char( jobz );
    char range_ = to_char( range );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ka_ = to_lapack_int( ka );
    lapack_int kb_ = to_lapack_int( kb );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int ldbb_ = to_lapack_int( ldbb );
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

    // allocate workspace
    lapack::vector< std::complex<float> > work( (n) );
    lapack::vector< float > rwork( (7*n) );
    lapack::vector< lapack_int > iwork( (5*n) );

    LAPACK_chbgvx(
        &jobz_, &range_, &uplo_, &n_, &ka_, &kb_,
        (lapack_complex_float*) AB, &ldab_,
        (lapack_complex_float*) BB, &ldbb_,
        (lapack_complex_float*) Q, &ldq_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        (lapack_complex_float*) Z, &ldz_,
        (lapack_complex_float*) &work[0],
        &rwork[0],
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
int64_t hbgvx(
    lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n, int64_t ka, int64_t kb,
    std::complex<double>* AB, int64_t ldab,
    std::complex<double>* BB, int64_t ldbb,
    std::complex<double>* Q, int64_t ldq, double vl, double vu, int64_t il, int64_t iu, double abstol,
    int64_t* nfound,
    double* W,
    std::complex<double>* Z, int64_t ldz,
    int64_t* ifail )
{
    char jobz_ = to_char( jobz );
    char range_ = to_char( range );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ka_ = to_lapack_int( ka );
    lapack_int kb_ = to_lapack_int( kb );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int ldbb_ = to_lapack_int( ldbb );
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

    // allocate workspace
    lapack::vector< std::complex<double> > work( (n) );
    lapack::vector< double > rwork( (7*n) );
    lapack::vector< lapack_int > iwork( (5*n) );

    LAPACK_zhbgvx(
        &jobz_, &range_, &uplo_, &n_, &ka_, &kb_,
        (lapack_complex_double*) AB, &ldab_,
        (lapack_complex_double*) BB, &ldbb_,
        (lapack_complex_double*) Q, &ldq_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        (lapack_complex_double*) Z, &ldz_,
        (lapack_complex_double*) &work[0],
        &rwork[0],
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
