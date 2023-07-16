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
void test_pocon_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t align = params.align();
    int64_t verbose = params.verbose();
    params.matrix.mark();

    real_t eps = std::numeric_limits< real_t >::epsilon();

    // mark non-standard output values
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();

    if (! run) {
        params.matrix.kind.set_default( "rand_dominant" );
        return;
    }

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, n ), align );
    real_t anorm;
    real_t rcond_tst;
    real_t rcond_ref;
    size_t size_A = (size_t) lda * n;

    std::vector< scalar_t > A( size_A );

    lapack::generate_matrix( params.matrix, n, n, &A[0], lda );

    anorm = lapack::lanhe( lapack::Norm::One, uplo, n, &A[0], lda );

    if (verbose >= 1) {
        printf( "\n"
                "A n %lld, lda %lld, Anorm %.2e\n",
                llong( n ), llong( lda ), anorm );
    }

    // factor A into LL^T
    int64_t info = lapack::potrf( uplo, n, &A[0], lda );
    if (info != 0) {
        fprintf( stderr, "lapack::potrf returned error %lld\n", llong( info ) );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::pocon( uplo, n, &A[0], lda, anorm, &rcond_tst );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::pocon returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::pocon( n );
    //params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_pocon( uplo2char(uplo), n, &A[0], lda, anorm, &rcond_ref );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_pocon returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += std::abs( rcond_tst - rcond_ref );
        params.error() = error;
        params.okay() = (error < 3*eps);
    }
}

// -----------------------------------------------------------------------------
void test_pocon( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_pocon_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_pocon_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_pocon_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_pocon_work< std::complex<double> >( params, run );
            break;
    }
}
