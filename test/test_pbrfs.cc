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
#include "cblas_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_pbrfs_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t kd = params.kd();
    int64_t nrhs = params.nrhs();
    int64_t align = params.align();
    int64_t verbose = params.verbose();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol() * eps;

    // mark non-standard output values
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( kd+1, align );
    int64_t ldafb = ldab;
    int64_t ldb = roundup( blas::max( 1, n ), align );
    int64_t ldx = ldb;
    size_t size_AB = (size_t) ldab * n;
    size_t size_AFB = size_AB;
    size_t size_B = (size_t) ldb * nrhs;
    size_t size_X = size_B;
    size_t size_ferr = (size_t) (nrhs);
    size_t size_berr = (size_t) (nrhs);

    std::vector< scalar_t > AB( size_AB );
    std::vector< scalar_t > AFB( size_AFB );
    std::vector< scalar_t > B( size_B );
    std::vector< scalar_t > X_tst( size_X );
    std::vector< scalar_t > X_ref( size_X );
    std::vector< real_t > ferr_tst( size_ferr );
    std::vector< real_t > ferr_ref( size_ferr );
    std::vector< real_t > berr_tst( size_berr );
    std::vector< real_t > berr_ref( size_berr );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB.size(), &AB[0] );
    int64_t iseed_B[4];
    std::copy( iseed, iseed+4, iseed_B );
    lapack::larnv( idist, iseed, B.size(), &B[0] );
    X_tst = B;

    // diagonally dominant -> positive definite
    if (uplo == lapack::Uplo::Upper) {
        for (int64_t j = 0; j < n; ++j) {
            AB[ kd + j*ldab ] += n;
        }
    }
    else { // lower
        for (int64_t j = 0; j < n; ++j) {
            AB[ j*ldab ] += n;
        }
    }

    AFB = AB;

    // Factor
    int64_t info = lapack::pbtrf( uplo, n, kd, &AFB[0], ldafb );
    if (info != 0) {
        fprintf( stderr, "lapack::pbtrf returned error %lld\n", llong( info ) );
    }

    // Solve in X_tst
    info = lapack::pbtrs ( uplo, n, kd, nrhs, &AFB[0], ldab, &X_tst[0], ldx );
    if (info != 0) {
        fprintf( stderr, "lapack::pbtrs returned error %lld\n", llong( info ) );
    }
    X_ref = X_tst;

    if (verbose >= 2) {
        printf( "A_factor = " ); print_matrix( kd+1, n, &AFB[0], ldafb );
        printf( "X = " ); print_matrix( n, nrhs, &X_tst[0], ldx );
    }


    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    // Refine solution in X_tst, using original AB and B, factored AFB.
    int64_t info_tst = lapack::pbrfs(
        uplo, n, kd, nrhs, &AB[0], ldab, &AFB[0], ldafb,
        &B[0], ldb, &X_tst[0], ldx, &ferr_tst[0], &berr_tst[0] );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::pbrfs returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::pbrfs( n, kd, nrhs );
    //params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "Xrfs = " ); print_matrix( n, nrhs, &X_tst[0], ldx );
        printf( "ferr = " ); print_vector( n, &ferr_tst[0], 1 );
        printf( "berr = " ); print_vector( n, &berr_tst[0], 1 );
    }

    if (params.check() == 'y') {
        // ---------- check error
        // Relative backwards error = ||b - Ax|| / (n * ||A|| * ||x||).
        // No hbmm, so loop over RHS.
        for (int64_t j = 0; j < nrhs; ++j) {
            // B_ref -= A * B_tst
            cblas_hbmv( CblasColMajor, cblas_uplo_const(uplo), n, kd,
                        -1.0, &AB[0], ldab,
                              &X_tst[ j*ldx ], 1,
                         1.0, &B[ j*ldb ], 1 );
        }
        if (verbose >= 2) {
            printf( "R = " ); print_matrix( n, nrhs, &B[0], ldb );
        }

        real_t error = lapack::lange( lapack::Norm::One, n, nrhs, &B[0], ldb );
        real_t Xnorm = lapack::lange( lapack::Norm::One, n, nrhs, &X_tst[0], ldx );
        real_t Anorm = lapack::lanhb( lapack::Norm::One, uplo, n, kd, &AB[0], ldab );
        error /= (n * Anorm * Xnorm);
        params.error() = error;
        params.okay() = (error < tol);

        // Reset B for ref using saved seed.
        lapack::larnv( idist, iseed_B, B.size(), &B[0] );
    }

    if (params.ref() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_pbrfs(
            uplo2char(uplo), n, kd, nrhs, &AB[0], ldab, &AFB[0], ldafb,
            &B[0], ldb, &X_ref[0], ldx, &ferr_ref[0], &berr_ref[0] );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_pbrfs returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_pbrfs( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_pbrfs_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_pbrfs_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_pbrfs_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_pbrfs_work< std::complex<double> >( params, run );
            break;
    }
}
