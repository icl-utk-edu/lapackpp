// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/device.hh"

namespace lapack {

using blas::max, blas::min;
using blas::conj;

template <typename scalar_t>
int64_t tpqrt2(
    int64_t m, int64_t n, int64_t l,
    scalar_t* dA, int64_t ldda,
    scalar_t* dB, int64_t lddb,
    scalar_t* dT, int64_t lddt,
    lapack::Queue& queue )
{
    #define dA(i_, j_) ( dA + i_ + (j_)*ldda )
    #define dB(i_, j_) ( dB + i_ + (j_)*lddb )
    #define dT(i_, j_) ( dT + i_ + (j_)*lddt )

    scalar_t one  = 1.0;
    scalar_t zero = 0.0;

    int64_t info = 0;
    if (m < 0)
       info = -1;
    else if (n < 0)
       info = -2;
    else if (l < 0 || l > min( m, n ))
       info = -3;
    else if (ldda < max( 1, n ))
       info = -5;
    else if (lddb < max( 1, m ))
       info = -7;
    else if (lddt < max( 1, n ))
       info = -9;

    if (info != 0)
       return info;

    // Quick return if possible
    if (m == 0 || n == 0)
       return 0;

    Op op_trans = (blas::is_complex_v< scalar_t > ? Op::ConjTrans : Op::Trans);

    for (int i = 0; i < n; ++i) {
        scalar_t* tau   = dT(i, 0);     // tau calculated from larfg
        scalar_t* A_row = dA(i, i+1);   // Remaining elements of current row of A to be transformed
        scalar_t* v     = dB(0, i);     // Block reflector from larfg
        scalar_t* B_rem = dB(0, i+1);   // Remaining block of B to be transformed
        scalar_t* work  = dT(0, n-1);   // Temporary workspace

        // Generate elementary reflector H(i) to annihilate B(:, i)
        int64_t p = m-l+min( l, i+1 );
        lapack::larfg( p+1, dA(i, i), v, 1, tau, queue );

        if (i < n-1) {
            // Apply block reflector to C

            // Compute v^H C_i where C_i = [ A_row ; B_rem ], (work = A_row^H + B^H v)
            blas::conj( n-i-1, A_row, ldda, work, 1, queue );
            blas::gemv( Layout::ColMajor, op_trans, p, n-i-1, one, B_rem, lddb, v, 1, one, work, 1, queue );

            // Apply H to A_row (A_row = A_row - tau * work^H)
            // alpha = -conj( tau )
            scalar_t alpha;
            blas::device_memcpy( &alpha, tau, 1, queue );
            queue.sync();
            alpha = -conj( alpha );
            // A_row += alpha * work^H for j = [0, n-i-1)
            scalar_t* temp = blas::device_malloc< scalar_t >( n-i-1, queue );   // Allocate intermediate temp vector
            blas::conj( n-i-1, work, 1, temp, 1, queue );
            blas::axpy( n-i-1, alpha, temp, 1, A_row, ldda, queue );
            queue.sync();
            blas::device_free( temp, queue );  // Free temp vector

            // Apply H to B
            // B_rem = B_rem + alpha*v*work^H
            blas::ger( Layout::ColMajor, p, n-i-1, alpha, v, 1, work, 1, B_rem, lddb, queue );
        }
    }
    for (int i = 1; i < n; ++i) {
        // Get T matrix

        // T(1:I-1,I) := C(I:M,1:I-1)**H * (alpha * C(I:M,I))

        // alpha = -dT(i, 0)
        scalar_t alpha;
        blas::device_memcpy( &alpha, dT(i, 0), 1, queue );
        queue.sync();
        alpha = -alpha;
        // dT(j, i) = zero for j = [0, i)
        blas::device_memset( dT(0, i), 0, i, queue );

        int64_t p = min( i, l );
        int64_t mp = min( m-l, m-1 );
        int64_t np = min( p, n-1 );

        // Triangular part of B2
        // T(j, i) = alpha * B(m-l, i)
        blas::device_memcpy( dT(0, i), dB(m-l, i), p, queue );
        blas::scal( p, alpha, dT(0, i), 1, queue );
        blas::trmv( Layout::ColMajor, Uplo::Upper, op_trans, Diag::NonUnit, p, dB(mp, 0), lddb, dT(0, i), 1, queue );

        // Rectangular part of B2
        blas::gemv( Layout::ColMajor, op_trans, l, i-p, alpha, dB(mp, np), lddb, dB(mp, i), 1, zero, dT(np, i), 1, queue );

        // B1
        blas::gemv( Layout::ColMajor, op_trans, m-l, i, alpha, dB, lddb, dB(0, i), 1, one, dT(0, i), 1, queue );

        // T(1:I-1,I) := T(1:I-1,1:I-1) * T(1:I-1,I)
        blas::trmv( Layout::ColMajor, Uplo::Upper, Op::NoTrans, Diag::NonUnit, i, dT, lddt, dT(0, i), 1, queue );

        // T(i, i) = tau
        blas::device_memcpy( dT(i, i), dT(i, 0), 1, queue );
        blas::device_memcpy( dT(i, 0), &zero, 1, queue );
    }

    return info;
}

template int64_t tpqrt2(
    int64_t m, int64_t n, int64_t l,
    float* dA, int64_t ldda,
    float* dB, int64_t lddb,
    float* dT, int64_t lddt,
    lapack::Queue& queue );

template int64_t tpqrt2(
    int64_t m, int64_t n, int64_t l,
    double* dA, int64_t ldda,
    double* dB, int64_t lddb,
    double* dT, int64_t lddt,
    lapack::Queue& queue );

template int64_t tpqrt2(
    int64_t m, int64_t n, int64_t l,
    std::complex<float>* dA, int64_t ldda,
    std::complex<float>* dB, int64_t lddb,
    std::complex<float>* dT, int64_t lddt,
    lapack::Queue& queue );

template int64_t tpqrt2(
    int64_t m, int64_t n, int64_t l,
    std::complex<double>* dA, int64_t ldda,
    std::complex<double>* dB, int64_t lddb,
    std::complex<double>* dT, int64_t lddt,
    lapack::Queue& queue );

}  // namespace lapack
