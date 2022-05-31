// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t stein(
    int64_t n,
    float const* D,
    float const* E, int64_t m,
    float const* W,
    int64_t const* iblock,
    int64_t const* isplit,
    float* Z, int64_t ldz,
    int64_t* ifail )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int m_ = (lapack_int) m;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > iblock_( &iblock[0], &iblock[(n)] );
        lapack_int const* iblock_ptr = &iblock_[0];
    #else
        lapack_int const* iblock_ptr = iblock;
    #endif
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > isplit_( &isplit[0], &isplit[(n)] );
        lapack_int const* isplit_ptr = &isplit_[0];
    #else
        lapack_int const* isplit_ptr = isplit;
    #endif
    lapack_int ldz_ = (lapack_int) ldz;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ifail_( (m) );
        lapack_int* ifail_ptr = &ifail_[0];
    #else
        lapack_int* ifail_ptr = ifail;
    #endif
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (5*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_sstein(
        &n_,
        D,
        E, &m_,
        W,
        iblock_ptr,
        isplit_ptr,
        Z, &ldz_,
        &work[0],
        &iwork[0],
        ifail_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ifail_.begin(), ifail_.end(), ifail );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t stein(
    int64_t n,
    double const* D,
    double const* E, int64_t m,
    double const* W,
    int64_t const* iblock,
    int64_t const* isplit,
    double* Z, int64_t ldz,
    int64_t* ifail )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int m_ = (lapack_int) m;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > iblock_( &iblock[0], &iblock[(n)] );
        lapack_int const* iblock_ptr = &iblock_[0];
    #else
        lapack_int const* iblock_ptr = iblock;
    #endif
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > isplit_( &isplit[0], &isplit[(n)] );
        lapack_int const* isplit_ptr = &isplit_[0];
    #else
        lapack_int const* isplit_ptr = isplit;
    #endif
    lapack_int ldz_ = (lapack_int) ldz;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ifail_( (m) );
        lapack_int* ifail_ptr = &ifail_[0];
    #else
        lapack_int* ifail_ptr = ifail;
    #endif
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (5*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_dstein(
        &n_,
        D,
        E, &m_,
        W,
        iblock_ptr,
        isplit_ptr,
        Z, &ldz_,
        &work[0],
        &iwork[0],
        ifail_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ifail_.begin(), ifail_.end(), ifail );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t stein(
    int64_t n,
    float const* D,
    float const* E, int64_t m,
    float const* W,
    int64_t const* iblock,
    int64_t const* isplit,
    std::complex<float>* Z, int64_t ldz,
    int64_t* ifail )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int m_ = (lapack_int) m;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > iblock_( &iblock[0], &iblock[(n)] );
        lapack_int const* iblock_ptr = &iblock_[0];
    #else
        lapack_int const* iblock_ptr = iblock;
    #endif
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > isplit_( &isplit[0], &isplit[(n)] );
        lapack_int const* isplit_ptr = &isplit_[0];
    #else
        lapack_int const* isplit_ptr = isplit;
    #endif
    lapack_int ldz_ = (lapack_int) ldz;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ifail_( (m) );
        lapack_int* ifail_ptr = &ifail_[0];
    #else
        lapack_int* ifail_ptr = ifail;
    #endif
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (5*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_cstein(
        &n_,
        D,
        E, &m_,
        W,
        iblock_ptr,
        isplit_ptr,
        (lapack_complex_float*) Z, &ldz_,
        &work[0],
        &iwork[0],
        ifail_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ifail_.begin(), ifail_.end(), ifail );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t stein(
    int64_t n,
    double const* D,
    double const* E, int64_t m,
    double const* W,
    int64_t const* iblock,
    int64_t const* isplit,
    std::complex<double>* Z, int64_t ldz,
    int64_t* ifail )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int m_ = (lapack_int) m;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > iblock_( &iblock[0], &iblock[(n)] );
        lapack_int const* iblock_ptr = &iblock_[0];
    #else
        lapack_int const* iblock_ptr = iblock;
    #endif
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > isplit_( &isplit[0], &isplit[(n)] );
        lapack_int const* isplit_ptr = &isplit_[0];
    #else
        lapack_int const* isplit_ptr = isplit;
    #endif
    lapack_int ldz_ = (lapack_int) ldz;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ifail_( (m) );
        lapack_int* ifail_ptr = &ifail_[0];
    #else
        lapack_int* ifail_ptr = ifail;
    #endif
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (5*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_zstein(
        &n_,
        D,
        E, &m_,
        W,
        iblock_ptr,
        isplit_ptr,
        (lapack_complex_double*) Z, &ldz_,
        &work[0],
        &iwork[0],
        ifail_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ifail_.begin(), ifail_.end(), ifail );
    #endif
    return info_;
}

}  // namespace lapack
