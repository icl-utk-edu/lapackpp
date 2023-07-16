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
void test_potrs_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t align = params.align();
    int64_t verbose = params.verbose();
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();
    params.gflops();

    if (! run) {
        params.matrix.kind.set_default( "rand_dominant" );
        return;
    }

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, n ), align );
    int64_t ldb = roundup( blas::max( 1, n ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_B = (size_t) ldb * nrhs;

    std::vector< scalar_t > A( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    lapack::generate_matrix( params.matrix, n, n, &A[0], lda );
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    B_ref = B_tst;

    // factor A into LL^T
    int64_t info = lapack::potrf( uplo, n, &A[0], lda );
    if (info != 0) {
        fprintf( stderr, "lapack::potrf returned error %lld\n", llong( info ) );
    }

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld\n"
                "B n=%5lld, nrhs=%5lld, ldb=%5lld",
                llong( n ), llong( lda ),
                llong( n ), llong( nrhs ), llong( ldb ) );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( n, n, &A[0], lda );
        printf( "B = " ); print_matrix( n, nrhs, &B_tst[0], lda );
    }

    // test error exits
    if (params.error_exit() == 'y') {
        using lapack::Uplo;
        assert_throw( lapack::potrs( Uplo(0),  n, nrhs, &A[0], lda, &B_tst[0], ldb ), lapack::Error );
        assert_throw( lapack::potrs( uplo,    -1, nrhs, &A[0], lda, &B_tst[0], ldb ), lapack::Error );
        assert_throw( lapack::potrs( uplo,     n,   -1, &A[0], lda, &B_tst[0], ldb ), lapack::Error );
        assert_throw( lapack::potrs( uplo,     n, nrhs, &A[0], n-1, &B_tst[0], ldb ), lapack::Error );
        assert_throw( lapack::potrs( uplo,     n, nrhs, &A[0], lda, &B_tst[0], n-1 ), lapack::Error );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::potrs( uplo, n, nrhs, &A[0], lda, &B_tst[0], ldb );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::potrs returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::potrs( n, nrhs );
    params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "B2 = " ); print_matrix( n, nrhs, &B_tst[0], ldb );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_potrs( uplo2char(uplo), n, nrhs, &A[0], lda, &B_ref[0], ldb );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_potrs returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "B2ref = " ); print_matrix( n, nrhs, &B_ref[0], ldb );
        }

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( B_tst, B_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_potrs( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_potrs_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_potrs_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_potrs_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_potrs_work< std::complex<double> >( params, run );
            break;
    }
}
