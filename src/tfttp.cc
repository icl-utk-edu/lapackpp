// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t tfttp(
    lapack::Op transr, lapack::Uplo uplo, int64_t n,
    float const* ARF,
    float* AP )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_stfttp(
        &transr_, &uplo_, &n_,
        ARF,
        AP, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tfttp(
    lapack::Op transr, lapack::Uplo uplo, int64_t n,
    double const* ARF,
    double* AP )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_dtfttp(
        &transr_, &uplo_, &n_,
        ARF,
        AP, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tfttp(
    lapack::Op transr, lapack::Uplo uplo, int64_t n,
    std::complex<float> const* ARF,
    std::complex<float>* AP )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_ctfttp(
        &transr_, &uplo_, &n_,
        (lapack_complex_float*) ARF,
        (lapack_complex_float*) AP, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tfttp(
    lapack::Op transr, lapack::Uplo uplo, int64_t n,
    std::complex<double> const* ARF,
    std::complex<double>* AP )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_ztfttp(
        &transr_, &uplo_, &n_,
        (lapack_complex_double*) ARF,
        (lapack_complex_double*) AP, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
