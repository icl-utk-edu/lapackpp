// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#ifdef LAPACK_HAVE_MATGEN

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t laghe(
    int64_t n, int64_t k,
    float const* D,
    std::complex<float>* A, int64_t lda,
    int64_t* iseed )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int k_ = to_lapack_int( k );
    lapack_int lda_ = to_lapack_int( lda );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > iseed_( &iseed[0], &iseed[(4)] );
        lapack_int* iseed_ptr = &iseed_[0];
    #else
        lapack_int* iseed_ptr = iseed;
    #endif
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (2*n) );

    LAPACK_claghe(
        &n_, &k_,
        D,
        (lapack_complex_float*) A, &lda_,
        iseed_ptr,
        (lapack_complex_float*) &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( iseed_.begin(), iseed_.end(), iseed );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t laghe(
    int64_t n, int64_t k,
    double const* D,
    std::complex<double>* A, int64_t lda,
    int64_t* iseed )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int k_ = to_lapack_int( k );
    lapack_int lda_ = to_lapack_int( lda );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > iseed_( &iseed[0], &iseed[(4)] );
        lapack_int* iseed_ptr = &iseed_[0];
    #else
        lapack_int* iseed_ptr = iseed;
    #endif
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (2*n) );

    LAPACK_zlaghe(
        &n_, &k_,
        D,
        (lapack_complex_double*) A, &lda_,
        iseed_ptr,
        (lapack_complex_double*) &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( iseed_.begin(), iseed_.end(), iseed );
    #endif
    return info_;
}

}  // namespace lapack

#endif  // LAPACK_HAVE_MATGEN
