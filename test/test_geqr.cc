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

#if LAPACK_VERSION >= 30700  // >= 3.7.0

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_geqr_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align();

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();
    params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, m ), align );
    size_t size_A = (size_t) lda * n;

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > T_tst( 5 );  // 5 is minimum

    // query for T size (pass tsize = -1 for optimal, tsize = -2 for minimum)
    int64_t info_tst = lapack::geqr( m, n, &A_tst[0], lda, &T_tst[0], -1 );
    if (info_tst != 0) {
        fprintf( stderr, "lapack::geqr query returned error %lld\n", llong( info_tst ) );
    }
    int64_t tsize = std::real( T_tst[0] );
    assert( tsize >= 5 );
    T_tst.resize( tsize );
    std::vector< scalar_t > T_ref( tsize );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    A_ref = A_tst;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    info_tst = lapack::geqr( m, n, &A_tst[0], lda, &T_tst[0], tsize );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::geqr returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::geqrf( m, n );
    params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_geqr( m, n, &A_ref[0], lda, &T_ref[0], tsize );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_geqr returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( T_tst, T_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_geqr( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_geqr_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_geqr_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_geqr_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_geqr_work< std::complex<double> >( params, run );
            break;
    }
}

#else

// -----------------------------------------------------------------------------
void test_geqr( Params& params, bool run )
{
    fprintf( stderr, "geqr requires LAPACK >= 3.7.0\n\n" );
    exit(0);
}

#endif  // LAPACK >= 3.7.0
