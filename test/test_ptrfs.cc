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
void test_ptrfs_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t align = params.align();

    // mark non-standard output values
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t ldb = roundup( blas::max( 1, n ), align );
    int64_t ldx = roundup( blas::max( 1, n ), align );
    size_t size_D = (size_t) (n);
    size_t size_E = (size_t) (n-1);
    size_t size_DF = (size_t) (n);
    size_t size_EF = (size_t) (n-1);
    size_t size_B = (size_t) ldb * nrhs;
    size_t size_X = (size_t) ldx * nrhs;
    size_t size_ferr = (size_t) (nrhs);
    size_t size_berr = (size_t) (nrhs);

    std::vector< real_t > D( size_D );
    std::vector< scalar_t > E( size_E );
    std::vector< real_t > DF( size_DF );
    std::vector< scalar_t > EF( size_EF );
    std::vector< scalar_t > B( size_B );
    std::vector< scalar_t > X_tst( size_X );
    std::vector< scalar_t > X_ref( size_X );
    std::vector< real_t > ferr_tst( size_ferr );
    std::vector< real_t > ferr_ref( size_ferr );
    std::vector< real_t > berr_tst( size_berr );
    std::vector< real_t > berr_ref( size_berr );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, D.size(), &D[0] );
    lapack::larnv( idist, iseed, E.size(), &E[0] );
    lapack::larnv( idist, iseed, DF.size(), &DF[0] );
    lapack::larnv( idist, iseed, EF.size(), &EF[0] );
    lapack::larnv( idist, iseed, B.size(), &B[0] );
    lapack::larnv( idist, iseed, X_tst.size(), &X_tst[0] );
    X_ref = X_tst;

    // diagonally dominant -> positive definite
    for (int64_t i = 0; i < n; ++i) {
        D[ i ] += n;
    }

    // factor using pttrf
    int64_t info = lapack::pttrf( n, &D[0], &E[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::pttrf returned error %lld\n", llong( info ) );
    }

    // solve using pttrs
    info = lapack::pttrs( uplo, n, nrhs, &D[0], &E[0], &B[0], ldb );
    if (info != 0) {
        fprintf( stderr, "lapack::pttrs returned error %lld\n", llong( info ) );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::ptrfs( uplo, n, nrhs, &D[0], &E[0], &DF[0], &EF[0], &B[0], ldb, &X_tst[0], ldx, &ferr_tst[0], &berr_tst[0] );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::ptrfs returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::ptrfs( n, nrhs );
    //params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_ptrfs( uplo2char(uplo), n, nrhs, &D[0], &E[0], &DF[0], &EF[0], &B[0], ldb, &X_ref[0], ldx, &ferr_ref[0], &berr_ref[0] );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_ptrfs returned error %lld\n", llong( info_ref ) );
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
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_ptrfs( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_ptrfs_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_ptrfs_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_ptrfs_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_ptrfs_work< std::complex<double> >( params, run );
            break;
    }
}
