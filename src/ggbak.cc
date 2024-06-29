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
int64_t ggbak(
    lapack::Balance balance, lapack::Side side, int64_t n, int64_t ilo, int64_t ihi,
    float const* lscale,
    float const* rscale, int64_t m,
    float* V, int64_t ldv )
{
    char balance_ = to_char( balance );
    char side_ = to_char( side );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ilo_ = to_lapack_int( ilo );
    lapack_int ihi_ = to_lapack_int( ihi );
    lapack_int m_ = to_lapack_int( m );
    lapack_int ldv_ = to_lapack_int( ldv );
    lapack_int info_ = 0;

    LAPACK_sggbak(
        &balance_, &side_, &n_, &ilo_, &ihi_,
        lscale,
        rscale, &m_,
        V, &ldv_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbak(
    lapack::Balance balance, lapack::Side side, int64_t n, int64_t ilo, int64_t ihi,
    double const* lscale,
    double const* rscale, int64_t m,
    double* V, int64_t ldv )
{
    char balance_ = to_char( balance );
    char side_ = to_char( side );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ilo_ = to_lapack_int( ilo );
    lapack_int ihi_ = to_lapack_int( ihi );
    lapack_int m_ = to_lapack_int( m );
    lapack_int ldv_ = to_lapack_int( ldv );
    lapack_int info_ = 0;

    LAPACK_dggbak(
        &balance_, &side_, &n_, &ilo_, &ihi_,
        lscale,
        rscale, &m_,
        V, &ldv_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbak(
    lapack::Balance balance, lapack::Side side, int64_t n, int64_t ilo, int64_t ihi,
    float const* lscale,
    float const* rscale, int64_t m,
    std::complex<float>* V, int64_t ldv )
{
    char balance_ = to_char( balance );
    char side_ = to_char( side );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ilo_ = to_lapack_int( ilo );
    lapack_int ihi_ = to_lapack_int( ihi );
    lapack_int m_ = to_lapack_int( m );
    lapack_int ldv_ = to_lapack_int( ldv );
    lapack_int info_ = 0;

    LAPACK_cggbak(
        &balance_, &side_, &n_, &ilo_, &ihi_,
        lscale,
        rscale, &m_,
        (lapack_complex_float*) V, &ldv_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbak(
    lapack::Balance balance, lapack::Side side, int64_t n, int64_t ilo, int64_t ihi,
    double const* lscale,
    double const* rscale, int64_t m,
    std::complex<double>* V, int64_t ldv )
{
    char balance_ = to_char( balance );
    char side_ = to_char( side );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ilo_ = to_lapack_int( ilo );
    lapack_int ihi_ = to_lapack_int( ihi );
    lapack_int m_ = to_lapack_int( m );
    lapack_int ldv_ = to_lapack_int( ldv );
    lapack_int info_ = 0;

    LAPACK_zggbak(
        &balance_, &side_, &n_, &ilo_, &ihi_,
        lscale,
        rscale, &m_,
        (lapack_complex_double*) V, &ldv_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
