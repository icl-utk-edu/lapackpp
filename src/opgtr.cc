// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t opgtr(
    lapack::Uplo uplo, int64_t n,
    float const* AP,
    float const* tau,
    float* Q, int64_t ldq )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int ldq_ = (lapack_int) ldq;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (n-1) );

    LAPACK_sopgtr(
        &uplo_, &n_,
        AP,
        tau,
        Q, &ldq_,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t opgtr(
    lapack::Uplo uplo, int64_t n,
    double const* AP,
    double const* tau,
    double* Q, int64_t ldq )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int ldq_ = (lapack_int) ldq;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (n-1) );

    LAPACK_dopgtr(
        &uplo_, &n_,
        AP,
        tau,
        Q, &ldq_,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
