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
int64_t steqr(
    lapack::Job compz, int64_t n,
    float* D,
    float* E,
    float* Z, int64_t ldz )
{
    char compz_ = to_char_comp( compz );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (max( 1, 2*n-2 )) );

    LAPACK_ssteqr(
        &compz_, &n_,
        D,
        E,
        Z, &ldz_,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t steqr(
    lapack::Job compz, int64_t n,
    double* D,
    double* E,
    double* Z, int64_t ldz )
{
    char compz_ = to_char_comp( compz );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (max( 1, 2*n-2 )) );

    LAPACK_dsteqr(
        &compz_, &n_,
        D,
        E,
        Z, &ldz_,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t steqr(
    lapack::Job compz, int64_t n,
    float* D,
    float* E,
    std::complex<float>* Z, int64_t ldz )
{
    char compz_ = to_char_comp( compz );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (max( 1, 2*n-2 )) );

    LAPACK_csteqr(
        &compz_, &n_,
        D,
        E,
        (lapack_complex_float*) Z, &ldz_,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t steqr(
    lapack::Job compz, int64_t n,
    double* D,
    double* E,
    std::complex<double>* Z, int64_t ldz )
{
    char compz_ = to_char_comp( compz );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (max( 1, 2*n-2 )) );

    LAPACK_zsteqr(
        &compz_, &n_,
        D,
        E,
        (lapack_complex_double*) Z, &ldz_,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
