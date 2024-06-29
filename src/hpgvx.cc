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
int64_t hpgvx(
    int64_t itype, lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n,
    std::complex<float>* AP,
    std::complex<float>* BP, float vl, float vu, int64_t il, int64_t iu, float abstol,
    int64_t* m,
    float* W,
    std::complex<float>* Z, int64_t ldz,
    int64_t* ifail )
{
    lapack_int itype_ = to_lapack_int( itype );
    char jobz_ = to_char( jobz );
    char range_ = to_char( range );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int il_ = to_lapack_int( il );
    lapack_int iu_ = to_lapack_int( iu );
    lapack_int m_ = to_lapack_int( *m );
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
    lapack::vector< std::complex<float> > work( (2*n) );
    lapack::vector< float > rwork( (7*n) );
    lapack::vector< lapack_int > iwork( (5*n) );

    LAPACK_chpgvx(
        &itype_, &jobz_, &range_, &uplo_, &n_,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) BP, &vl, &vu, &il_, &iu_, &abstol, &m_,
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
    *m = m_;
    #ifndef LAPACK_ILP64
        if (jobz != Job::NoVec) {
            std::copy( &ifail_[ 0 ], &ifail_[ m_ ], ifail );
        }
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t hpgvx(
    int64_t itype, lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n,
    std::complex<double>* AP,
    std::complex<double>* BP, double vl, double vu, int64_t il, int64_t iu, double abstol,
    int64_t* m,
    double* W,
    std::complex<double>* Z, int64_t ldz,
    int64_t* ifail )
{
    lapack_int itype_ = to_lapack_int( itype );
    char jobz_ = to_char( jobz );
    char range_ = to_char( range );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int il_ = to_lapack_int( il );
    lapack_int iu_ = to_lapack_int( iu );
    lapack_int m_ = to_lapack_int( *m );
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
    lapack::vector< std::complex<double> > work( (2*n) );
    lapack::vector< double > rwork( (7*n) );
    lapack::vector< lapack_int > iwork( (5*n) );

    LAPACK_zhpgvx(
        &itype_, &jobz_, &range_, &uplo_, &n_,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) BP, &vl, &vu, &il_, &iu_, &abstol, &m_,
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
    *m = m_;
    #ifndef LAPACK_ILP64
        if (jobz != Job::NoVec) {
            std::copy( &ifail_[ 0 ], &ifail_[ m_ ], ifail );
        }
    #endif
    return info_;
}

}  // namespace lapack
