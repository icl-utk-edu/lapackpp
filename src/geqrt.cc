// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#if LAPACK_VERSION >= 30400  // >= 3.4

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t geqrt(
    int64_t m, int64_t n, int64_t nb,
    float* A, int64_t lda,
    float* T, int64_t ldt )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nb_ = to_lapack_int( nb );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (nb*n) );

    LAPACK_sgeqrt(
        &m_, &n_, &nb_,
        A, &lda_,
        T, &ldt_,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geqrt(
    int64_t m, int64_t n, int64_t nb,
    double* A, int64_t lda,
    double* T, int64_t ldt )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nb_ = to_lapack_int( nb );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (nb*n) );

    LAPACK_dgeqrt(
        &m_, &n_, &nb_,
        A, &lda_,
        T, &ldt_,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geqrt(
    int64_t m, int64_t n, int64_t nb,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* T, int64_t ldt )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nb_ = to_lapack_int( nb );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (nb*n) );

    LAPACK_cgeqrt(
        &m_, &n_, &nb_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) T, &ldt_,
        (lapack_complex_float*) &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geqrt(
    int64_t m, int64_t n, int64_t nb,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* T, int64_t ldt )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nb_ = to_lapack_int( nb );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (nb*n) );

    LAPACK_zgeqrt(
        &m_, &n_, &nb_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) T, &ldt_,
        (lapack_complex_double*) &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.4
