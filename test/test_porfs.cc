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
void test_porfs_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t align = params.align();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();

    if (! run) {
        params.matrix.kind.set_default( "rand_dominant" );
        return;
    }

    // ---------- setup
    // make A and AF, B and X, the same size
    int64_t lda = roundup( blas::max( 1, n ), align );
    int64_t ldaf = lda;
    int64_t ldb = roundup( blas::max( 1, n ), align );
    int64_t ldx = ldb;
    size_t size_A = (size_t) lda * n;
    size_t size_AF = (size_t) ldaf * n;
    size_t size_B = (size_t) ldb * nrhs;
    size_t size_X = (size_t) ldx * nrhs;
    size_t size_ferr = (size_t) (nrhs);
    size_t size_berr = (size_t) (nrhs);

    std::vector< scalar_t > A( size_A );
    std::vector< scalar_t > AF( size_AF );
    std::vector< scalar_t > B( size_B );
    std::vector< scalar_t > X_tst( size_X );
    std::vector< scalar_t > X_ref( size_X );
    std::vector< real_t > ferr_tst( size_ferr );
    std::vector< real_t > ferr_ref( size_ferr );
    std::vector< real_t > berr_tst( size_berr );
    std::vector< real_t > berr_ref( size_berr );

    lapack::generate_matrix( params.matrix, n, n, &A[0], lda );
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, B.size(), &B[0] );

    // factor AF = LU
    AF = A;
    X_tst = B;
    int64_t info = lapack::potrf( uplo, n, &AF[0], lda );
    if (info != 0) {
        fprintf( stderr, "lapack::potrf returned error %lld\n", llong( info ) );
    }

    // initial solve of AF X = B
    info = lapack::potrs( uplo, n, nrhs, &AF[0], lda, &X_tst[0], ldx );
    if (info != 0) {
        fprintf( stderr, "lapack::potrs returned error %lld\n", llong( info ) );
    }
    X_ref = X_tst;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::porfs( uplo, n, nrhs, &A[0], lda, &AF[0], ldaf, &B[0], ldb, &X_tst[0], ldx, &ferr_tst[0], &berr_tst[0] );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::porfs returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::porfs( n, nrhs );
    //params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_porfs( uplo2char(uplo), n, nrhs, &A[0], lda, &AF[0], ldaf, &B[0], ldb, &X_ref[0], ldx, &ferr_ref[0], &berr_ref[0] );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_porfs returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( X_tst, X_ref );
        error += abs_error( ferr_tst, ferr_ref );
        error += abs_error( berr_tst, berr_ref );
        params.error() = error;
        params.okay() = (error < 3*eps);
    }
}

// -----------------------------------------------------------------------------
void test_porfs( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_porfs_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_porfs_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_porfs_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_porfs_work< std::complex<double> >( params, run );
            break;
    }
}
