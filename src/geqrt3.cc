// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"

#if LAPACK_VERSION >= 30400  // >= 3.4

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t geqrt3(
    int64_t m, int64_t n,
    float* A, int64_t lda,
    float* T, int64_t ldt )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int info_ = 0;

    LAPACK_sgeqrt3(
        &m_, &n_,
        A, &lda_,
        T, &ldt_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geqrt3(
    int64_t m, int64_t n,
    double* A, int64_t lda,
    double* T, int64_t ldt )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int info_ = 0;

    LAPACK_dgeqrt3(
        &m_, &n_,
        A, &lda_,
        T, &ldt_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geqrt3(
    int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* T, int64_t ldt )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int info_ = 0;

    LAPACK_cgeqrt3(
        &m_, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) T, &ldt_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geqrt3(
    int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* T, int64_t ldt )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int info_ = 0;

    LAPACK_zgeqrt3(
        &m_, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) T, &ldt_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.4
