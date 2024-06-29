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
int64_t tftri(
    lapack::Op transr, lapack::Uplo uplo, lapack::Diag diag, int64_t n,
    float* A )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    char diag_ = to_char( diag );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_stftri(
        &transr_, &uplo_, &diag_, &n_,
        A, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tftri(
    lapack::Op transr, lapack::Uplo uplo, lapack::Diag diag, int64_t n,
    double* A )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    char diag_ = to_char( diag );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_dtftri(
        &transr_, &uplo_, &diag_, &n_,
        A, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tftri(
    lapack::Op transr, lapack::Uplo uplo, lapack::Diag diag, int64_t n,
    std::complex<float>* A )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    char diag_ = to_char( diag );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_ctftri(
        &transr_, &uplo_, &diag_, &n_,
        (lapack_complex_float*) A, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tftri(
    lapack::Op transr, lapack::Uplo uplo, lapack::Diag diag, int64_t n,
    std::complex<double>* A )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    char diag_ = to_char( diag );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_ztftri(
        &transr_, &uplo_, &diag_, &n_,
        (lapack_complex_double*) A, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
