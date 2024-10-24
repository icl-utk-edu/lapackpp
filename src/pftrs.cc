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
int64_t pftrs(
    lapack::Op transr, lapack::Uplo uplo, int64_t n, int64_t nrhs,
    float const* A,
    float* B, int64_t ldb )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_spftrs(
        &transr_, &uplo_, &n_, &nrhs_,
        A,
        B, &ldb_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t pftrs(
    lapack::Op transr, lapack::Uplo uplo, int64_t n, int64_t nrhs,
    double const* A,
    double* B, int64_t ldb )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_dpftrs(
        &transr_, &uplo_, &n_, &nrhs_,
        A,
        B, &ldb_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t pftrs(
    lapack::Op transr, lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<float> const* A,
    std::complex<float>* B, int64_t ldb )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_cpftrs(
        &transr_, &uplo_, &n_, &nrhs_,
        (lapack_complex_float*) A,
        (lapack_complex_float*) B, &ldb_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t pftrs(
    lapack::Op transr, lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<double> const* A,
    std::complex<double>* B, int64_t ldb )
{
    char transr_ = to_char( transr );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_zpftrs(
        &transr_, &uplo_, &n_, &nrhs_,
        (lapack_complex_double*) A,
        (lapack_complex_double*) B, &ldb_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
