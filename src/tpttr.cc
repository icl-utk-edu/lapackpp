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
int64_t tpttr(
    lapack::Uplo uplo, int64_t n,
    float const* AP,
    float* A, int64_t lda )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_stpttr(
        &uplo_, &n_,
        AP,
        A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tpttr(
    lapack::Uplo uplo, int64_t n,
    double const* AP,
    double* A, int64_t lda )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_dtpttr(
        &uplo_, &n_,
        AP,
        A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tpttr(
    lapack::Uplo uplo, int64_t n,
    std::complex<float> const* AP,
    std::complex<float>* A, int64_t lda )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_ctpttr(
        &uplo_, &n_,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tpttr(
    lapack::Uplo uplo, int64_t n,
    std::complex<double> const* AP,
    std::complex<double>* A, int64_t lda )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_ztpttr(
        &uplo_, &n_,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
