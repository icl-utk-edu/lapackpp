// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "lapack.hh"
#include "lapack/device.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <limits>
#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_unmqr_device_work( Params& params, bool run )
{
    using lapack::device_info_int;
    using real_t = blas::real_type< scalar_t >;
    using blas::max;

    // get & mark input values
    lapack::Side side = params.side();
    lapack::Op trans = params.trans();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
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
    params.msg();

    if (! run)
        return;

    // Q is mq-by-mq, the product of k reflectors.
    int64_t mq = (side == lapack::Side::Left ? m : n);

    // skip invalid sizes and options
    if (k > mq) {
        params.msg() = "skipping: requires k <= m (left) or k <= n (right)";
        return;
    }
    if (blas::is_complex_v< scalar_t > && trans == lapack::Op::Trans) {
        params.msg() = "skipping: complex requires trans = n or c";
        return;
    }

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // ---------- setup
    int64_t lda = roundup( max( 1, mq ), align );
    int64_t ldc = roundup( max( 1, m ), align );
    size_t size_A   = (size_t) max( 1, lda * k );
    size_t size_tau = (size_t) max( 1, k );
    size_t size_C   = (size_t) max( 1, ldc * n );

    std::vector< scalar_t > A( size_A );
    std::vector< scalar_t > tau( size_tau );
    std::vector< scalar_t > C_tst( size_C );
    std::vector< scalar_t > C_ref( size_C );

    // Householder reflectors from the QR factorization of an mq-by-k matrix.
    lapack::generate_matrix( params.matrix, mq, k, A.data(), lda );
    int64_t info_qrf = lapack::geqrf( mq, k, A.data(), lda, tau.data() );
    if (info_qrf != 0) {
        fprintf( stderr, "lapack::geqrf returned error %lld\n", llong( info_qrf ) );
    }

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, C_tst.size(), C_tst.data() );
    C_ref = C_tst;

    // Allocate and copy to GPU.
    lapack::Queue queue( device );
    scalar_t*        dA     = blas::device_malloc< scalar_t >( size_A, queue );
    scalar_t*        d_tau  = blas::device_malloc< scalar_t >( size_tau, queue );
    scalar_t*        dC_tst = blas::device_malloc< scalar_t >( size_C, queue );
    device_info_int* d_info = blas::device_malloc< device_info_int >( 1, queue );
    blas::device_copy_matrix( mq, k, A.data(), lda, dA, lda, queue );
    blas::device_memcpy( d_tau, tau.data(), k, queue );
    blas::device_copy_matrix( m, n, C_tst.data(), ldc, dC_tst, ldc, queue );

    // Set info to nonzero, to check that unmqr sets it.
    device_info_int info_tst = -1;
    blas::device_memcpy( d_info, &info_tst, 1, queue );

    if (verbose >= 1) {
        printf( "\n"
                "A mq=%5lld, k=%5lld, lda=%5lld\n"
                "C m=%5lld, n=%5lld, ldc=%5lld\n",
                llong( mq ), llong( k ), llong( lda ),
                llong( m ), llong( n ), llong( ldc ) );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( mq, k, A.data(), lda );
        printf( "C = " ); print_matrix( m, n, C_tst.data(), ldc );
    }

    // Allocate workspace.
    // Real uses the ormqr name, complex unmqr; both are the same routine.
    size_t d_size, h_size;
    if constexpr (blas::is_complex_v< scalar_t >) {
        lapack::unmqr_work_size_bytes( side, trans, m, n, k, dA, lda, d_tau,
                                       dC_tst, ldc, &d_size, &h_size, queue );
    }
    else {
        lapack::ormqr_work_size_bytes( side, trans, m, n, k, dA, lda, d_tau,
                                       dC_tst, ldc, &d_size, &h_size, queue );
    }
    char* d_work = blas::device_malloc< char >( d_size, queue );
    std::vector<char> h_work_vector( h_size );
    char* h_work = h_work_vector.data();

    // test error exits
    if (params.error_exit() == 'y') {
        if (blas::is_complex_v< scalar_t >) {
            // complex does trans = n, c, not t
            assert_throw( lapack::unmqr( side, lapack::Op::Trans, m, n, k, dA, lda, d_tau, dC_tst, ldc, d_work, d_size, h_work, h_size, d_info, queue ), lapack::Error );
        }
        #if defined(LAPACK_HAVE_CUBLAS) || defined(LAPACK_HAVE_ROCBLAS)
            // cuSolver and rocSolver take 32-bit int sizes.
            int64_t big = int64_t( std::numeric_limits<int>::max() ) + 1;
            assert_throw( lapack::unmqr( side, trans, m, n, k, dA, lda, d_tau, dC_tst, big, d_work, d_size, h_work, h_size, d_info, queue ), lapack::Error );
        #endif
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    queue.sync();
    double time = testsweeper::get_wtime();

    if constexpr (blas::is_complex_v< scalar_t >) {
        lapack::unmqr( side, trans, m, n, k, dA, lda, d_tau, dC_tst, ldc,
                       d_work, d_size, h_work, h_size, d_info, queue );
    }
    else {
        lapack::ormqr( side, trans, m, n, k, dA, lda, d_tau, dC_tst, ldc,
                       d_work, d_size, h_work, h_size, d_info, queue );
    }

    queue.sync();
    time = testsweeper::get_wtime() - time;

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::unmqr( side, m, n, k );
    params.gflops() = gflop / time;

    // Copy result back to CPU.
    blas::device_copy_matrix( m, n, dC_tst, ldc, C_tst.data(), ldc, queue );
    blas::device_memcpy( &info_tst, d_info, 1, queue );
    queue.sync();

    if (info_tst != 0) {
        fprintf( stderr, "lapack::unmqr returned error %lld\n", llong( info_tst ) );
    }

    // Cleanup GPU memory.
    blas::device_free( dA, queue );
    blas::device_free( d_tau, queue );
    blas::device_free( dC_tst, queue );
    blas::device_free( d_info, queue );
    blas::device_free( d_work, queue );

    if (verbose >= 2) {
        printf( "C_out = " ); print_matrix( m, n, C_tst.data(), ldc );
    }

    if (params.check() == 'y' || params.ref() == 'y') {
        real_t Cnorm = lapack::lange( lapack::Norm::One, m, n, C_ref.data(), ldc );

        // ---------- run reference, LAPACK++ on the host
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = lapack::unmqr( side, trans, m, n, k, A.data(), lda,
                                          tau.data(), C_ref.data(), ldc );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "lapack::unmqr (host) returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "C_ref = " ); print_matrix( m, n, C_ref.data(), ldc );
        }

        // ---------- check error compared to reference
        // || C_tst - C_ref ||_1 / (mq || C ||_1); Q is unitary.
        blas::axpy( size_C, -1.0, C_ref.data(), 1, C_tst.data(), 1 );
        real_t error = lapack::lange( lapack::Norm::One, m, n, C_tst.data(), ldc );
        if (Cnorm > 0)
            error /= (mq * Cnorm);

        params.error() = error;
        params.okay() = (error < tol) && (info_tst == 0) && (info_ref == 0);
    }
}

// -----------------------------------------------------------------------------
void test_unmqr_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_unmqr_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_unmqr_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_unmqr_device_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_unmqr_device_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
