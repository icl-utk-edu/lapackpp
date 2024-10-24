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
int64_t sbgv(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t ka, int64_t kb,
    float* AB, int64_t ldab,
    float* BB, int64_t ldbb,
    float* W,
    float* Z, int64_t ldz )
{
    char jobz_ = to_char( jobz );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ka_ = to_lapack_int( ka );
    lapack_int kb_ = to_lapack_int( kb );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int ldbb_ = to_lapack_int( ldbb );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (3*n) );

    LAPACK_ssbgv(
        &jobz_, &uplo_, &n_, &ka_, &kb_,
        AB, &ldab_,
        BB, &ldbb_,
        W,
        Z, &ldz_,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t sbgv(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t ka, int64_t kb,
    double* AB, int64_t ldab,
    double* BB, int64_t ldbb,
    double* W,
    double* Z, int64_t ldz )
{
    char jobz_ = to_char( jobz );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ka_ = to_lapack_int( ka );
    lapack_int kb_ = to_lapack_int( kb );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int ldbb_ = to_lapack_int( ldbb );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (3*n) );

    LAPACK_dsbgv(
        &jobz_, &uplo_, &n_, &ka_, &kb_,
        AB, &ldab_,
        BB, &ldbb_,
        W,
        Z, &ldz_,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
