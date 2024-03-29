// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#if LAPACK_VERSION >= 30400  // >= 3.4

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup tpqrt
int64_t tpqrt(
    int64_t m, int64_t n, int64_t l, int64_t nb,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    float* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(l) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int l_ = (lapack_int) l;
    lapack_int nb_ = (lapack_int) nb;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (nb*n) );

    LAPACK_stpqrt(
        &m_, &n_, &l_, &nb_,
        A, &lda_,
        B, &ldb_,
        T, &ldt_,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup tpqrt
int64_t tpqrt(
    int64_t m, int64_t n, int64_t l, int64_t nb,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    double* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(l) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int l_ = (lapack_int) l;
    lapack_int nb_ = (lapack_int) nb;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (nb*n) );

    LAPACK_dtpqrt(
        &m_, &n_, &l_, &nb_,
        A, &lda_,
        B, &ldb_,
        T, &ldt_,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup tpqrt
int64_t tpqrt(
    int64_t m, int64_t n, int64_t l, int64_t nb,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(l) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int l_ = (lapack_int) l;
    lapack_int nb_ = (lapack_int) nb;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (nb*n) );

    LAPACK_ctpqrt(
        &m_, &n_, &l_, &nb_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) T, &ldt_,
        (lapack_complex_float*) &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup tpqrt
int64_t tpqrt(
    int64_t m, int64_t n, int64_t l, int64_t nb,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(l) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int l_ = (lapack_int) l;
    lapack_int nb_ = (lapack_int) nb;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (nb*n) );

    LAPACK_ztpqrt(
        &m_, &n_, &l_, &nb_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) T, &ldt_,
        (lapack_complex_double*) &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.4
