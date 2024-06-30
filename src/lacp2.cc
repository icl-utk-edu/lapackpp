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
void lacp2(
    lapack::Uplo uplo, int64_t m, int64_t n,
    float const* A, int64_t lda,
    std::complex<float>* B, int64_t ldb )
{
    char uplo_ = to_char( uplo );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );

    LAPACK_clacp2(
        &uplo_, &m_, &n_,
        A, &lda_,
        (lapack_complex_float*) B, &ldb_
    );
}

// -----------------------------------------------------------------------------
void lacp2(
    lapack::Uplo uplo, int64_t m, int64_t n,
    double const* A, int64_t lda,
    std::complex<double>* B, int64_t ldb )
{
    char uplo_ = to_char( uplo );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );

    LAPACK_zlacp2(
        &uplo_, &m_, &n_,
        A, &lda_,
        (lapack_complex_double*) B, &ldb_
    );
}

}  // namespace lapack
