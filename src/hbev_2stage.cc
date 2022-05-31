// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#if LAPACK_VERSION >= 30700  // >= 3.7

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t hbev_2stage(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<float>* AB, int64_t ldab,
    float* W,
    std::complex<float>* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int ldz_ = (lapack_int) ldz;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_chbev_2stage(
        &jobz_, &uplo_, &n_, &kd_,
        (lapack_complex_float*) AB, &ldab_,
        W,
        (lapack_complex_float*) Z, &ldz_,
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
    lapack::vector< float > rwork( (max( 1, 3*n-2 )) );

    LAPACK_chbev_2stage(
        &jobz_, &uplo_, &n_, &kd_,
        (lapack_complex_float*) AB, &ldab_,
        W,
        (lapack_complex_float*) Z, &ldz_,
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
int64_t hbev_2stage(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<double>* AB, int64_t ldab,
    double* W,
    std::complex<double>* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int ldz_ = (lapack_int) ldz;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_zhbev_2stage(
        &jobz_, &uplo_, &n_, &kd_,
        (lapack_complex_double*) AB, &ldab_,
        W,
        (lapack_complex_double*) Z, &ldz_,
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
    lapack::vector< double > rwork( (max( 1, 3*n-2 )) );

    LAPACK_zhbev_2stage(
        &jobz_, &uplo_, &n_, &kd_,
        (lapack_complex_double*) AB, &ldab_,
        W,
        (lapack_complex_double*) Z, &ldz_,
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

#endif  // LAPACK >= 3.7
