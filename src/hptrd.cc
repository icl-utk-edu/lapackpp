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
int64_t hptrd(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* AP,
    float* D,
    float* E,
    std::complex<float>* tau )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_chptrd(
        &uplo_, &n_,
        (lapack_complex_float*) AP,
        D,
        E,
        (lapack_complex_float*) tau, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t hptrd(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* AP,
    double* D,
    double* E,
    std::complex<double>* tau )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_zhptrd(
        &uplo_, &n_,
        (lapack_complex_double*) AP,
        D,
        E,
        (lapack_complex_double*) tau, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
