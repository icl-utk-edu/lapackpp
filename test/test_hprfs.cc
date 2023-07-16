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
void test_hprfs_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t align = params.align();

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t ldb = roundup( blas::max( 1, n ), align );
    int64_t ldx = roundup( blas::max( 1, n ), align );
    size_t size_AP = (size_t) (n*(n+1)/2);
    size_t size_AFP = (size_t) (n*(n+1)/2);
    size_t size_ipiv = (size_t) (n);
    size_t size_B = (size_t) ldb * nrhs;
    size_t size_X = (size_t) ldx * nrhs;
    size_t size_ferr = (size_t) (nrhs);
    size_t size_berr = (size_t) (nrhs);

    std::vector< scalar_t > AP( size_AP );
    std::vector< scalar_t > AFP( size_AFP );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );
    std::vector< scalar_t > B( size_B );
    std::vector< scalar_t > X_tst( size_X );
    std::vector< scalar_t > X_ref( size_X );
    std::vector< real_t > ferr_tst( size_ferr );
    std::vector< real_t > ferr_ref( size_ferr );
    std::vector< real_t > berr_tst( size_berr );
    std::vector< real_t > berr_ref( size_berr );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AP.size(), &AP[0] );
    AFP = AP;
    lapack::larnv( idist, iseed, B.size(), &B[0] );
    X_tst = B;

    // factor to initialize ipiv_tst
    int64_t info_trf = lapack::hptrf( uplo, n, &AFP[0], &ipiv_tst[0] );
    if (info_trf != 0) {
        fprintf( stderr, "lapack::hptrf returned error %lld\n", llong( info_trf ) );
    }

    // solve
    int64_t info_trs = lapack::hptrs( uplo, n, nrhs, &AFP[0], &ipiv_tst[0], &X_tst[0], ldb );
    if (info_trs != 0) {
        fprintf( stderr, "lapack::hptrs returned error %lld\n", llong( info_trs ) );
    }

    // Use results from lapackpp for the reference LAPACKE run
    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );
    X_ref = X_tst;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::hprfs( uplo, n, nrhs, &AP[0], &AFP[0], &ipiv_tst[0], &B[0], ldb, &X_tst[0], ldx, &ferr_tst[0], &berr_tst[0] );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::hprfs returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::hprfs( n, nrhs );
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_hprfs( uplo2char(uplo), n, nrhs, &AP[0], &AFP[0], &ipiv_ref[0], &B[0], ldb, &X_ref[0], ldx, &ferr_ref[0], &berr_ref[0] );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_hprfs returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        real_t eps = std::numeric_limits< real_t >::epsilon();
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( X_tst, X_ref );
        error += abs_error( ferr_tst, ferr_ref );
        error += abs_error( berr_tst, berr_ref );
        params.error() = error;
        params.okay() = (error < 3*eps);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_hprfs( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_hprfs_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_hprfs_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_hprfs_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_hprfs_work< std::complex<double> >( params, run );
            break;
    }
}
