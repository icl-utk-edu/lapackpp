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
int64_t sbtrd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t kd,
    float* AB, int64_t ldab,
    float* D,
    float* E,
    float* Q, int64_t ldq )
{
    char jobz_ = to_char( jobz );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (n) );

    LAPACK_ssbtrd(
        &jobz_, &uplo_, &n_, &kd_,
        AB, &ldab_,
        D,
        E,
        Q, &ldq_,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t sbtrd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t kd,
    double* AB, int64_t ldab,
    double* D,
    double* E,
    double* Q, int64_t ldq )
{
    char jobz_ = to_char( jobz );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (n) );

    LAPACK_dsbtrd(
        &jobz_, &uplo_, &n_, &kd_,
        AB, &ldab_,
        D,
        E,
        Q, &ldq_,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
