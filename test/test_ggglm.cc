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
void test_ggglm_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t m = params.dim.m();
    int64_t p = params.dim.k();
    int64_t align = params.align();
    params.matrix.mark();
    params.matrixB.mark();

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();
    params.msg();

    if (! run)
        return;

    // skip invalid sizes
    if (! (p >= n-m)) {
        params.msg() = "skipping: requires p >= n-m";
        return;
    }
    if (! (m <= n)) {
        params.msg() = "skipping: requires m <= n";
        return;
    }

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, n ), align );
    int64_t ldb = roundup( blas::max( 1, n ), align );
    size_t size_A = (size_t) ( lda * m );
    size_t size_B = (size_t) ( ldb * p );
    size_t size_D = (size_t) (n);
    size_t size_X = (size_t) (m);
    size_t size_Y = (size_t) (p);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );
    std::vector< scalar_t > D_tst( size_D );
    std::vector< scalar_t > D_ref( size_D );
    std::vector< scalar_t > X_tst( size_X );
    std::vector< scalar_t > X_ref( size_X );
    std::vector< scalar_t > Y_tst( size_Y );
    std::vector< scalar_t > Y_ref( size_Y );

    lapack::generate_matrix( params.matrix,  n, m, &A_tst[0], lda );
    lapack::generate_matrix( params.matrixB, n, p, &B_tst[0], ldb );
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, D_tst.size(), &D_tst[0] );
    A_ref = A_tst;
    B_ref = B_tst;
    D_ref = D_tst;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::ggglm( n, m, p, &A_tst[0], lda, &B_tst[0], ldb, &D_tst[0], &X_tst[0], &Y_tst[0] );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::ggglm returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::ggglm( n, m );
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_ggglm( n, m, p, &A_ref[0], lda, &B_ref[0], ldb, &D_ref[0], &X_ref[0], &Y_ref[0] );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_ggglm returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( B_tst, B_ref );
        error += abs_error( D_tst, D_ref );
        error += abs_error( X_tst, X_ref );
        error += abs_error( Y_tst, Y_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_ggglm( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_ggglm_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_ggglm_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_ggglm_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_ggglm_work< std::complex<double> >( params, run );
            break;
    }
}
