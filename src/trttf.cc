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
int64_t trttf(
    lapack::Op transr, lapack::Uplo uplo, int64_t n,
    float const* A, int64_t lda,
    float* ARF )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_strttf(
        &transr_, &uplo_, &n_,
        A, &lda_,
        ARF, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t trttf(
    lapack::Op transr, lapack::Uplo uplo, int64_t n,
    double const* A, int64_t lda,
    double* ARF )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_dtrttf(
        &transr_, &uplo_, &n_,
        A, &lda_,
        ARF, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t trttf(
    lapack::Op transr, lapack::Uplo uplo, int64_t n,
    std::complex<float> const* A, int64_t lda,
    std::complex<float>* ARF )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_ctrttf(
        &transr_, &uplo_, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) ARF, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t trttf(
    lapack::Op transr, lapack::Uplo uplo, int64_t n,
    std::complex<double> const* A, int64_t lda,
    std::complex<double>* ARF )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_ztrttf(
        &transr_, &uplo_, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) ARF, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
