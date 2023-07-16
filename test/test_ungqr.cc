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
void test_ungqr_work( Params& params, bool run )
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
    if (! (n <= m && k <= n)) {
        params.msg() = "skipping: requires n <= m and k <= n";
        return;
    }

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, m ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_tau = (size_t) (k);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > A_factored( size_A );
    std::vector< scalar_t > tau( size_tau );

    lapack::generate_matrix( params.matrix, m, n, &A_tst[0], lda );
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, tau.size(), &tau[0] );
    // save matrix
    A_ref = A_tst;

    // ---------- factor matrix
    int64_t info_qrf = lapack::geqrf( m, n, &A_tst[0], lda, &tau[0] );
    if (info_qrf != 0) {
        fprintf( stderr, "lapack::unqrf returned error %lld\n", llong( info_qrf ) );
    }
    // save matrix
    A_factored = A_tst;

    // // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::ungqr( m, n, k, &A_tst[0], lda, &tau[0] );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::ungqr returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::ungqr( m, n, k );
    params.gflops() = gflop / time;

    if (params.check() == 'y') {
        // ---------- check error
        // comparing to ref. solution doesn't work
        // Following lapack/TESTING/LIN/zqrt02.f
        // Note (0 <= n <= m)  (0 <= k <= n).

        int64_t ldq = blas::max( m, n );
        int64_t ldr = blas::max( m, n );
        std::vector< scalar_t > Q( ldq * n );
        std::vector< scalar_t > R( ldr * n );

        // Copy the first k columns of the factorization to the array Q
        real_t rogue = -10000000000; // -1D+10
        lapack::laset( lapack::MatrixType::General, m, n, rogue, rogue, &Q[0], ldq );
        lapack::lacpy( lapack::MatrixType::Lower, m-1, k, &A_factored[1], lda, &Q[1], ldq );

        // Generate the first n columns of the matrix Q
        int64_t info_ungqr = lapack::ungqr( m, n, k, &Q[0], ldq, &tau[0] );
        if (info_ungqr != 0) {
            fprintf( stderr, "lapack::ungqr returned error %lld\n", llong( info_ungqr ) );
        }

        // Copy R(1:n,1:k)
        lapack::laset( lapack::MatrixType::General, n, k, 0.0, 0.0, &R[0], ldr );
        lapack::lacpy( lapack::MatrixType::Upper, n, k, &A_factored[0], lda, &R[0], ldr );

        // Compute R - Q'*A
        blas::gemm( blas::Layout::ColMajor,
                    blas::Op::ConjTrans, blas::Op::NoTrans, n, k, m,
                    -1.0, &Q[0], ldq, &A_ref[0], lda, 1.0, &R[0], ldr );

        // Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) .
        real_t Anorm = lapack::lange( lapack::Norm::One, m, k, &A_ref[0], lda );
        real_t resid1 = lapack::lange( lapack::Norm::One, n, k, &R[0], ldr );
        real_t error1 = 0;
        if (Anorm > 0)
            error1 = resid1 / ( m * Anorm );

        // Compute I - Q'*Q
        lapack::laset( lapack::MatrixType::General, n, n, 0.0, 1.0, &R[0], ldr );
        blas::herk( blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::ConjTrans,
                    n, m, -1.0, &Q[0], ldq, 1.0, &R[0], ldr );

        // Compute norm( I - Q'*Q ) / ( M * EPS ) .
        real_t resid2 = lapack::lansy( lapack::Norm::One, lapack::Uplo::Upper, n, &R[0], ldr );
        real_t error2 = ( resid2 / m );

        params.error() = error1;
        params.ortho() = error2;
        params.okay() = (error1 < tol) && (error2 < tol);
    }

    if (params.ref() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_ungqr( m, n, k, &A_ref[0], lda, &tau[0] );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_ungqr returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_ungqr( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_ungqr_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_ungqr_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_ungqr_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_ungqr_work< std::complex<double> >( params, run );
            break;
    }
}
