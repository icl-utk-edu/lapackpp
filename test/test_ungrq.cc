// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "lapack.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_ungrq_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    int64_t align = params.align();
    params.matrix.mark();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol() * eps;

    // mark non-standard output values
    params.ortho();
    params.gflops();
    params.ref_time();
    params.ref_gflops();
    params.msg();

    if (! run)
        return;

    // skip invalid sizes
    if (! (n >= m && m >= k)) {
        params.msg() = "skipping: requires n >= m and m >= k";
        return;
    }

    // ---------- setup
    // For error check, R needs to be k-by-m;
    // for ortho check, R needs to be m-by-m to store Q*Q^H.
    // zrqt02.f has R to be m-by-n, which is bigger than needed.
    int64_t lda = roundup( blas::max( 1, m ), align );
    int64_t ldr = lda;
    size_t size_A   = (size_t) lda * n;
    size_t size_tau = (size_t) (k);
    size_t size_R   = (size_t) lda * m;

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > tau( size_tau );
    std::vector< scalar_t > R( size_R );

    lapack::generate_matrix( params.matrix, m, n, &A_tst[0], lda );
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, tau.size(), &tau[0] );
    A_ref = A_tst;

    // ---------- factor matrix
    // A is m-by-n, but factor bottom k-by-n portion.
    // (Compare with QR, which would factor the left m-by-k portion.)
    int64_t info_rqf = lapack::gerqf( k, n, &A_tst[(m-k)], lda, &tau[0] );
    if (info_rqf != 0) {
        fprintf( stderr, "lapack::gerqf returned error %lld\n", llong( info_rqf ) );
    }
    // Copy R(m-k+1:m, 1:n) from factored A for check.
    lapack::laset( lapack::MatrixType::General, k, m, 0.0, 0.0,
                   &R[(m-k)], ldr );
    lapack::lacpy( lapack::MatrixType::Upper,   k, k,
                   &A_tst[(m-k) + (n-k)*lda], lda,
                       &R[(m-k) + (m-k)*ldr], ldr );

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::ungrq( m, n, k, &A_tst[0], lda, &tau[0] );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::ungrq returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::ungrq( m, n, k );
    params.gflops() = gflop / time;

    if (params.check() == 'y') {
        // ---------- check error
        // comparing to ref. solution doesn't work
        // Following lapack/TESTING/LIN/zrqt02.f
        // Note: n >= m;  m >= k; lda >= m

        // Compute R(m-k+1:m, 1:n) - A(m-k+1:m, 1:n) * Q(n-m+1:n, 1:n)^H
        blas::gemm( blas::Layout::ColMajor,
                    blas::Op::NoTrans, blas::Op::ConjTrans, k, m, n,
                    -1.0, &A_ref[(m-k)], lda,
                          &A_tst[0],     lda,
                     1.0,     &R[(m-k)], ldr );

        // Compute norm( R - Q^H*A ) / ( M * norm(A) * EPS ) .
        real_t Anorm = lapack::lange( lapack::Norm::One, k, n, &A_ref[(m-k)], lda );
        real_t error = lapack::lange( lapack::Norm::One, k, m,     &R[(m-k)], ldr );
        if (Anorm > 0)
            error /= n * Anorm;

        // Compute I - Q*Q^H
        // Note Q has orthonormal rows (instead of cols).
        lapack::laset( lapack::MatrixType::General, m, m, 0.0, 1.0, &R[0], ldr );
        blas::herk( blas::Layout::ColMajor,
                    blas::Uplo::Upper, blas::Op::NoTrans, m, n,
                    -1.0, &A_tst[0], lda,
                     1.0,     &R[0], ldr );

        // Compute norm( I - Q*Q^H ) / ( N * EPS )
        real_t ortho = lapack::lansy( lapack::Norm::One, lapack::Uplo::Upper, m, &R[0], ldr );
        ortho /= n;

        params.error() = error;
        params.ortho() = ortho;
        params.okay() = (error < tol) && (ortho < tol);
    }

    if (params.ref() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_ungrq( m, n, k, &A_ref[0], lda, &tau[0] );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_ungrq returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_ungrq( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_ungrq_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_ungrq_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_ungrq_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_ungrq_work< std::complex<double> >( params, run );
            break;
    }
}
