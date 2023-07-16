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
void test_geqrf_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align();
    params.matrix.mark();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol() * eps;

    // mark non-standard output values
    //params.ref_time();
    //params.ref_gflops();
    params.gflops();
    params.ortho();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, m ), align );
    size_t size_A = (size_t)( lda * n );
    size_t size_tau = (size_t)( blas::min( m, n ) );
    int64_t minmn = blas::min( m, n );

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > tau_tst( size_tau );
    std::vector< scalar_t > tau_ref( size_tau );

    lapack::generate_matrix( params.matrix, m, n, &A_tst[0], lda );
    A_ref = A_tst;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::geqrf( m, n, &A_tst[0], lda, &tau_tst[0] );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::geqrf returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::geqrf( m, n );
    params.gflops() = gflop / time;

    if (params.check() == 'y') {
        // ---------- check error
        // comparing to ref. solution doesn't work
        // Following lapack/TESTING/LIN/zqrt01.f but using smaller Q and R
        int64_t ldq = m;
        std::vector< scalar_t > Q( m * minmn ); // m by k
        int64_t ldr = minmn;
        std::vector< scalar_t > R( minmn * n ); // k by n

        // Copy details of Q
        real_t rogue = -10000000000; // -1D+10
        lapack::laset( lapack::MatrixType::General, m, minmn, rogue, rogue, &Q[0], ldq );
        lapack::lacpy( lapack::MatrixType::Lower, m, minmn, &A_tst[0], lda, &Q[0], ldq );

        // Generate the m-by-m matrix Q
        int64_t info_ungqr = lapack::ungqr( m, minmn, minmn, &Q[0], ldq, &tau_tst[0] );
        if (info_ungqr != 0) {
            fprintf( stderr, "lapack::ungqr returned error %lld\n", llong( info_ungqr ) );
        }

        // Copy R
        lapack::laset( lapack::MatrixType::Lower, minmn, n, 0.0, 0.0, &R[0], ldr );
        lapack::lacpy( lapack::MatrixType::Upper, minmn, n, &A_tst[0], lda, &R[0], ldr );

        // Compute R - Q'*A
        blas::gemm( blas::Layout::ColMajor,
                    blas::Op::ConjTrans, blas::Op::NoTrans, minmn, n, m,
                    -1.0, &Q[0], ldq, &A_ref[0], lda, 1.0, &R[0], ldr );

        // Compute norm( R - Q'*A ) / ( M * norm(A) * EPS )
        real_t Anorm = lapack::lange( lapack::Norm::One, m, n, &A_ref[0], lda );
        real_t resid1 = lapack::lange( lapack::Norm::One, minmn, n, &R[0], ldr );
        real_t error1 = 0;
        if (Anorm > 0)
            error1 = resid1 / ( n * Anorm );

        // Compute I - Q'*Q
        lapack::laset( lapack::MatrixType::Upper, minmn, minmn, 0.0, 1.0, &R[0], ldr );
        blas::herk( blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::ConjTrans,
                    minmn, m, -1.0, &Q[0], ldq, 1.0, &R[0], ldr );

        // Compute norm( I - Q'*Q ) / ( M * EPS ) .
        real_t resid2 = lapack::lanhe( lapack::Norm::One, lapack::Uplo::Upper, minmn, &R[0], ldr );
        real_t error2 = ( resid2 / n );

        params.error() = error1;
        params.ortho() = error2;
        params.okay() = (error1 < tol) && (error2 < tol);
    }

    if (params.ref() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_geqrf( m, n, &A_ref[0], lda, &tau_ref[0] );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_geqrf returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_geqrf( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_geqrf_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_geqrf_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_geqrf_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_geqrf_work< std::complex<double> >( params, run );
            break;
    }
}
