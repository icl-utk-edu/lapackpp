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
int64_t tbrfs(
    lapack::Uplo uplo, lapack::Op trans, lapack::Diag diag, int64_t n, int64_t kd, int64_t nrhs,
    float const* AB, int64_t ldab,
    float const* B, int64_t ldb,
    float const* X, int64_t ldx,
    float* ferr,
    float* berr )
{
    char uplo_ = to_char( uplo );
    char trans_ = to_char( trans );
    char diag_ = to_char( diag );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (3*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_stbrfs(
        &uplo_, &trans_, &diag_, &n_, &kd_, &nrhs_,
        AB, &ldab_,
        B, &ldb_,
        X, &ldx_,
        ferr,
        berr,
        &work[0],
        &iwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tbrfs(
    lapack::Uplo uplo, lapack::Op trans, lapack::Diag diag, int64_t n, int64_t kd, int64_t nrhs,
    double const* AB, int64_t ldab,
    double const* B, int64_t ldb,
    double const* X, int64_t ldx,
    double* ferr,
    double* berr )
{
    char uplo_ = to_char( uplo );
    char trans_ = to_char( trans );
    char diag_ = to_char( diag );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (3*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_dtbrfs(
        &uplo_, &trans_, &diag_, &n_, &kd_, &nrhs_,
        AB, &ldab_,
        B, &ldb_,
        X, &ldx_,
        ferr,
        berr,
        &work[0],
        &iwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tbrfs(
    lapack::Uplo uplo, lapack::Op trans, lapack::Diag diag, int64_t n, int64_t kd, int64_t nrhs,
    std::complex<float> const* AB, int64_t ldab,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float> const* X, int64_t ldx,
    float* ferr,
    float* berr )
{
    char uplo_ = to_char( uplo );
    char trans_ = to_char( trans );
    char diag_ = to_char( diag );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (2*n) );
    lapack::vector< float > rwork( (n) );

    LAPACK_ctbrfs(
        &uplo_, &trans_, &diag_, &n_, &kd_, &nrhs_,
        (lapack_complex_float*) AB, &ldab_,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) X, &ldx_,
        ferr,
        berr,
        (lapack_complex_float*) &work[0],
        &rwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tbrfs(
    lapack::Uplo uplo, lapack::Op trans, lapack::Diag diag, int64_t n, int64_t kd, int64_t nrhs,
    std::complex<double> const* AB, int64_t ldab,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double> const* X, int64_t ldx,
    double* ferr,
    double* berr )
{
    char uplo_ = to_char( uplo );
    char trans_ = to_char( trans );
    char diag_ = to_char( diag );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldx_ = to_lapack_int( ldx );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (2*n) );
    lapack::vector< double > rwork( (n) );

    LAPACK_ztbrfs(
        &uplo_, &trans_, &diag_, &n_, &kd_, &nrhs_,
        (lapack_complex_double*) AB, &ldab_,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) X, &ldx_,
        ferr,
        berr,
        (lapack_complex_double*) &work[0],
        &rwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
