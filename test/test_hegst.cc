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
void test_hegst_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t itype = params.itype();
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t align = params.align();
    int64_t verbose = params.verbose();
    params.matrix.mark();
    params.matrixB.mark();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol() * eps;

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();

    if (! run) {
        params.matrixB.kind.set_default( "rand_dominant" );
        return;
    }

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, n ), align );
    int64_t ldb = roundup( blas::max( 1, n ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_B = (size_t) ldb * n;

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B( size_B );

    lapack::generate_matrix( params.matrix,  n, n, &A_tst[0], lda );
    lapack::generate_matrix( params.matrixB, n, n, &B[0], ldb );
    A_ref = A_tst;

    if (verbose >= 1) {
        printf( "\n" );
        printf( "A n=%5lld, lda=%5lld\n", llong( n ), llong( lda ) );
        printf( "B n=%5lld, ldb=%5lld\n", llong( n ), llong( ldb ) );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( n, n, &A_tst[0], lda );
        printf( "B = " ); print_matrix( n, n, &B[0], ldb );
    }

    // factor B
    int64_t info = lapack::potrf( uplo, n, &B[0], ldb );
    if (info != 0) {
        fprintf( stderr, "lapack::potrf returned error %lld\n", llong( info ) );
    }

    if (verbose >= 2) {
        printf( "Bhat = " ); print_matrix( n, n, &B[0], ldb );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::hegst(
        itype, uplo, n, &A_tst[0], lda, &B[0], ldb );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::hegst returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::hegst( itype, n );
    // params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "Ahat = " ); print_matrix( n, n, &A_tst[0], lda );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_hegst(
            itype, uplo2char(uplo), n, &A_ref[0], lda, &B[0], ldb );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_hegst returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        // relative forward error = ||A_ref - A_tst|| / ||A_ref||
        real_t Anorm = lapack::lanhe( lapack::Norm::One, uplo, n, &A_ref[0], lda );
        blas::axpy( size_A, -1.0, &A_tst[0], 1, &A_ref[0], 1 );
        real_t error = lapack::lanhe( lapack::Norm::One, uplo, n, &A_ref[0], lda );
        error /= Anorm;
        params.error() = error;
        params.okay() = (error < tol);
    }
}

// -----------------------------------------------------------------------------
void test_hegst( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_hegst_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_hegst_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_hegst_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_hegst_work< std::complex<double> >( params, run );
            break;
    }
}
