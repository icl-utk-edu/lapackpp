// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "lapack.hh"
#include "lapack/device.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_geqrf_device_work( Params& params, bool run )
{
    using lapack::device_info_int;
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t device = params.device();
    int64_t align = params.align();
    int64_t verbose = params.verbose();
    params.matrix.mark();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol() * eps;

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();
    params.gflops();
    params.ortho();

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, m ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_tau = (size_t) blas::min( m, n );
    int64_t minmn = blas::min( m, n );

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > tau_tst( size_tau );
    std::vector< scalar_t > tau_ref( size_tau );

    lapack::generate_matrix( params.matrix, m, n, &A_tst[0], lda );
    A_ref = A_tst;

    // Allocate and copy to GPU.
    lapack::Queue queue( device );
    scalar_t*        dA_tst = blas::device_malloc< scalar_t >( size_A, queue );
    scalar_t*        d_tau  = blas::device_malloc< scalar_t >( size_tau, queue );
    device_info_int* d_info = blas::device_malloc< device_info_int >( 1, queue );
    blas::device_copy_matrix( m, n, A_tst.data(), lda, dA_tst, lda, queue );

    if (verbose >= 1) {
        printf( "\n"
                "A m=%5lld, n=%5lld, lda=%5lld\n",
                llong( m ), llong( n ), llong( lda ) );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( m, n, &A_tst[0], lda );
    }

    // Allocate workspace
    size_t d_size, h_size;
    lapack::geqrf_work_size_bytes( m, n, dA_tst, lda, &d_size, &h_size, queue );
    char* d_work = blas::device_malloc< char >( d_size, queue );
    std::vector<char> h_work_vector( h_size );
    char* h_work = h_work_vector.data();

    // test error exits
    if (params.error_exit() == 'y') {
        assert_throw( lapack::geqrf( -1,  n, dA_tst, lda, d_tau, d_work, d_size, h_work, h_size, d_info, queue ), lapack::Error );
        assert_throw( lapack::geqrf(  m, -1, dA_tst, lda, d_tau, d_work, d_size, h_work, h_size, d_info, queue ), lapack::Error );
        assert_throw( lapack::geqrf(  m,  n, dA_tst, m-1, d_tau, d_work, d_size, h_work, h_size, d_info, queue ), lapack::Error );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    queue.sync();
    double time = testsweeper::get_wtime();

    lapack::geqrf( m, n, dA_tst, lda, d_tau,
                   d_work, d_size, h_work, h_size, d_info, queue );

    queue.sync();
    time = testsweeper::get_wtime() - time;

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::geqrf( m, n );
    params.gflops() = gflop / time;

    // Copy result back to CPU.
    device_info_int info_tst;
    blas::device_copy_matrix( m, n, dA_tst, lda, A_tst.data(), lda, queue );
    blas::device_memcpy( &info_tst, d_info, 1, queue );
    blas::device_memcpy( &tau_tst[0], d_tau, size_tau, queue );
    queue.sync();

    if (info_tst != 0) {
        fprintf( stderr, "lapack::geqrf returned error %lld\n", llong( info_tst ) );
    }

    // Cleanup GPU memory.
    blas::device_free( dA_tst, queue );
    blas::device_free( d_tau, queue  );
    blas::device_free( d_info, queue );
    blas::device_free( d_work, queue );

    if (verbose >= 2) {
        printf( "A_factor = " ); print_matrix( m, n, &A_tst[0], lda );
        printf( "tau = " ); print_matrix( 1, minmn, &tau_tst[0], 1 );
    }

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

        if (verbose >= 2) {
            printf( "Aref_factor = " ); print_matrix( m, n, &A_ref[0], lda );
        }
    }
}

// -----------------------------------------------------------------------------
void test_geqrf_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_geqrf_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_geqrf_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_geqrf_device_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_geqrf_device_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
