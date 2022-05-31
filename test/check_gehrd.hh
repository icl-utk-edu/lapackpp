// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas.hh"
#include "lapack.hh"

#include <vector>

// -----------------------------------------------------------------------------
// Given original A and factored H as the output of gehrd, computes:
// results[0] = || A - U H U^H || / (n ||A||)
// results[1] = || I - U U^H || / n
template< typename scalar_t >
void check_gehrd(
    int64_t n,
    scalar_t const* A, int64_t lda,
    scalar_t const* H, int64_t ldh,
    scalar_t const* tau,
    int64_t verbose,
    blas::real_type< scalar_t > results[2] )
{
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;

    size_t size_A = lda * n;
    std::vector< scalar_t > U( size_A ), H2( size_A ), work( size_A );

    // zero out below sub-diagonal in Hessenberg H2
    lapack::lacpy( lapack::MatrixType::General, n, n, H, lda, &H2[0], lda );
    lapack::laset( lapack::MatrixType::Lower, n-2, n-2, 0.0, 0.0, &H2[2], lda );
    // generate U
    lapack::lacpy( lapack::MatrixType::General, n, n, H, lda, &U[0], lda );
    lapack::unghr( n, 1, n, &U[0], lda, tau );
    // work = U H2
    blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, n, n,
                1.0, &U[0], lda,
                     &H2[0], lda,
                0.0, &work[0], lda );
    // H2 = A - (U H2) U^H
    lapack::lacpy( lapack::MatrixType::General, n, n, A, lda, &H2[0], lda );
    blas::gemm( Layout::ColMajor, Op::NoTrans, Op::ConjTrans, n, n, n,
                -1.0, &work[0], lda,
                      &U[0], lda,
                 1.0, &H2[0], lda );
    real_t Anorm = lapack::lange( lapack::Norm::One, n, n, &A[0], lda );
    real_t error = lapack::lange( lapack::Norm::One, n, n, &H2[0], lda );
    results[0] = error / Anorm / n;

    // work = I - U U^H
    lapack::laset( lapack::MatrixType::General, n, n, 0.0, 1.0, &work[0], lda );
    blas::gemm( Layout::ColMajor, Op::NoTrans, Op::ConjTrans, n, n, n,
                -1.0, &U[0], lda,
                      &U[0], lda,
                 1.0, &work[0], lda );
    error = lapack::lange( lapack::Norm::One, n, n, &work[0], lda );
    results[1] = error / n;
}
