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
int64_t hpgst(
    int64_t itype, lapack::Uplo uplo, int64_t n,
    std::complex<float>* AP,
    std::complex<float> const* BP )
{
    lapack_int itype_ = to_lapack_int( itype );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_chpgst(
        &itype_, &uplo_, &n_,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) BP, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t hpgst(
    int64_t itype, lapack::Uplo uplo, int64_t n,
    std::complex<double>* AP,
    std::complex<double> const* BP )
{
    lapack_int itype_ = to_lapack_int( itype );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_zhpgst(
        &itype_, &uplo_, &n_,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) BP, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
