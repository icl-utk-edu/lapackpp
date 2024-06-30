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
int64_t spgv(
    int64_t itype, lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    float* AP,
    float* BP,
    float* W,
    float* Z, int64_t ldz )
{
    lapack_int itype_ = to_lapack_int( itype );
    char jobz_ = to_char( jobz );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (3*n) );

    LAPACK_sspgv(
        &itype_, &jobz_, &uplo_, &n_,
        AP,
        BP,
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
int64_t spgv(
    int64_t itype, lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    double* AP,
    double* BP,
    double* W,
    double* Z, int64_t ldz )
{
    lapack_int itype_ = to_lapack_int( itype );
    char jobz_ = to_char( jobz );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (3*n) );

    LAPACK_dspgv(
        &itype_, &jobz_, &uplo_, &n_,
        AP,
        BP,
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
