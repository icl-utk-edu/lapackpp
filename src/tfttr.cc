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
int64_t tfttr(
    lapack::Op transr, lapack::Uplo uplo, int64_t n,
    float const* ARF,
    float* A, int64_t lda )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_stfttr(
        &transr_, &uplo_, &n_,
        ARF,
        A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tfttr(
    lapack::Op transr, lapack::Uplo uplo, int64_t n,
    double const* ARF,
    double* A, int64_t lda )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_dtfttr(
        &transr_, &uplo_, &n_,
        ARF,
        A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tfttr(
    lapack::Op transr, lapack::Uplo uplo, int64_t n,
    std::complex<float> const* ARF,
    std::complex<float>* A, int64_t lda )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_ctfttr(
        &transr_, &uplo_, &n_,
        (lapack_complex_float*) ARF,
        (lapack_complex_float*) A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tfttr(
    lapack::Op transr, lapack::Uplo uplo, int64_t n,
    std::complex<double> const* ARF,
    std::complex<double>* A, int64_t lda )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_ztfttr(
        &transr_, &uplo_, &n_,
        (lapack_complex_double*) ARF,
        (lapack_complex_double*) A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
