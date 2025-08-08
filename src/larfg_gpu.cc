// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/device.hh"
#include <cmath>

namespace lapack {

using std::real, std::imag, std::hypot;

template <typename real_t>
real_t hypot( std::complex<real_t> x, real_t y)
{
    return hypot( x.real(), x.imag(), y );
}

// -----------------------------------------------------------------------------
/// @ingroup larfg
template <typename scalar_t>
void larfg(
    int64_t n,
    scalar_t* alpha,
    scalar_t* dx, int64_t incdx,
    scalar_t* tau,
    lapack::Queue& queue )
{
    using real_t = blas::real_type< scalar_t >;

    const scalar_t one  = 1.0;

    // Quick return if n <= 0
    if (n <= 0) {
        blas::device_memset( tau, 0, 1, queue );
        return;
    }

    scalar_t alpha_;
    blas::device_memcpy( &alpha_, alpha, 1, queue );
    queue.sync();

    real_t xnorm;
    blas::nrm2( n-1, dx, incdx, &xnorm, queue );

    if (xnorm == 0 && imag(alpha_) == 0) {
        // h = i
        blas::device_memset( tau, 0, 1, queue );
        return;
    }

    // general case
    real_t beta = -copysign( hypot( alpha_, xnorm ), real(alpha_) );
    real_t safmin = std::numeric_limits<real_t>::min();
    real_t rsafmn = 1.0 / safmin;

    int64_t knt = 0;
    if (abs( beta ) < safmin) {
        // XNORM, BETA may be inaccurate; scale X and recompute them
        do {
            knt += 1;
            blas::scal( n-1, rsafmn, dx, incdx, queue );
            beta *= rsafmn;
            alpha_ *= rsafmn;
        } while (abs(beta) < safmin && knt < 20);

        blas::nrm2( n-1, dx, incdx, &xnorm, queue );
        beta = -copysign( hypot( alpha_, xnorm ), real(alpha_) );
    }

    scalar_t tau_ = (beta - alpha_) / beta;
    blas::device_memcpy( tau, &tau_, 1, queue );
    blas::scal( n-1, one / (alpha_ - beta), dx, incdx, queue );

    // If alpha is subnormal, it may lose relative accuracy

    for (int j = 0; j < knt; j++) {
        beta *= safmin;
    }
    alpha_ = beta;
    blas::device_memcpy( alpha, &alpha_, 1, queue );
}

template void larfg(
    int64_t n,
    float* alpha,
    float* dx, int64_t incdx,
    float* tau,
    lapack::Queue& queue );

template void larfg(
    int64_t n,
    double* alpha,
    double* dx, int64_t incdx,
    double* tau,
    lapack::Queue& queue );

template void larfg(
    int64_t n,
    std::complex<float>* alpha,
    std::complex<float>* dx, int64_t incdx,
    std::complex<float>* tau,
    lapack::Queue& queue );

template void larfg(
    int64_t n,
    std::complex<double>* alpha,
    std::complex<double>* dx, int64_t incdx,
    std::complex<double>* tau,
    lapack::Queue& queue );

}  // namespace lapack
