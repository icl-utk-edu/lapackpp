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
#include "lapacke_wrappers.hh"

#include <vector>

#if LAPACK_VERSION >= 30400  // >= 3.4.0

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_tprfb_device_work( Params& params, bool run )
{
    using lapack::device_info_int;
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Side side = params.side();
    lapack::Op trans = params.trans();
    lapack::Direction direction = params.direction();
    lapack::StoreV storev = params.storev();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    int64_t l = params.l();
    int64_t device = params.device();
    int64_t align = params.align();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol() * eps;

    // mark non-standard output values
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();
    params.msg();

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // skip invalid sizes
    if (k < l) {
        params.msg() = "skipping: requires k >= l";
        return;
    }

    // ---------- setup
    // B is m-by-n
    // V is m-by-k (left,  columnwise)
    //   or n-by-k (right, columnwise)
    //   or k-by-m (left,  rowwise)
    //   or k-by-n (right, rowwise)
    // T is k-by-k
    // A is k-by-n (left)
    //   or m-by-k (right)
    int64_t Vm, Vn;
    if (storev == lapack::StoreV::Columnwise) {
        Vm = (side == blas::Side::Left ? m : n);
        Vn = k;
    }
    else {
        Vm = k;
        Vn = (side == blas::Side::Left ? m : n);
    }
    int64_t Am = (side == blas::Side::Left ? k : m);
    int64_t An = (side == blas::Side::Left ? n : k);
    int64_t ldv = roundup( blas::max( 1, Vm ), align );
    int64_t ldt = roundup( blas::max( 1, k  ), align );
    int64_t lda = roundup( blas::max( 1, Am ), align );
    int64_t ldb = roundup( blas::max( 1, m  ), align );
    size_t size_V = (size_t) ldv * Vn;
    size_t size_T = (size_t) ldt * k;
    size_t size_A = (size_t) lda * An;
    size_t size_B = (size_t) ldb * n;

    std::vector< scalar_t > V( size_V );
    std::vector< scalar_t > T( size_T );
    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, V.size(), &V[0] );
    lapack::larnv( idist, iseed, T.size(), &T[0] );
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    A_ref = A_tst;
    B_ref = B_tst;

    // Allocate and copy to GPU.
    lapack::Queue queue( device );
    scalar_t* dV = blas::device_malloc< scalar_t >( size_V, queue );
    scalar_t* dT = blas::device_malloc< scalar_t >( size_T, queue );
    scalar_t* dA_tst = blas::device_malloc< scalar_t >( size_A, queue );
    scalar_t* dB_tst = blas::device_malloc< scalar_t >( size_B, queue );
    blas::device_copy_matrix( Vm, Vn, V.data(), ldv, dV, ldv, queue );
    blas::device_copy_matrix( k,  k,  T.data(), ldt, dT, ldt, queue );
    blas::device_copy_matrix( Am, An, A_tst.data(), lda, dA_tst, lda, queue );
    blas::device_copy_matrix( m,  n,  B_tst.data(), ldb, dB_tst, ldb, queue );

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    queue.sync();
    double time = testsweeper::get_wtime();
    lapack::tprfb( side, trans, direction, storev, m, n, k, l, dV, ldv, dT, ldt, dA_tst, lda, dB_tst, ldb, queue );
    queue.sync();
    time = testsweeper::get_wtime() - time;
    // internal routine: no argument check so no info.
    //if (info_tst != 0) {
    //    fprintf( stderr, "lapack::tprfb returned error %lld\n", llong( info_tst ) );
    //}

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::larfb( side, trans, direction, storev, m, n, k );
    //params.gflops() = gflop / time;

    // Copy result back to CPU
    blas::device_copy_matrix( Am, An, dA_tst, lda, A_tst.data(), lda, queue );
    blas::device_copy_matrix( m,  n,  dB_tst, ldb, B_tst.data(), ldb, queue );
    queue.sync();
    blas::device_free( dV, queue );
    blas::device_free( dT, queue );
    blas::device_free( dA_tst, queue );
    blas::device_free( dB_tst, queue );

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_tprfb( to_char( side ), to_char( trans ), to_char( direction ), to_char( storev ), m, n, k, l, &V[0], ldv, &T[0], ldt, &A_ref[0], lda, &B_ref[0], ldb );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_tprfb returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        //if (info_tst != info_ref) {
        //    error = 1;
        //}
        error += rel_error( A_tst, A_ref );
        error += rel_error( B_tst, B_ref );
        params.error() = error;
        params.okay() = (error < tol);
    }
}

// -----------------------------------------------------------------------------
void test_tprfb_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_tprfb_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_tprfb_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_tprfb_device_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_tprfb_device_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}

#else

// -----------------------------------------------------------------------------
void test_tprfb_device( Params& params, bool run )
{
    fprintf( stderr, "tprfb requires LAPACK >= 3.4.0\n\n" );
    exit(0);
}

#endif  // LAPACK >= 3.4.0
