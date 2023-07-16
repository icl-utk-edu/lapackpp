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
void test_gbtrs_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Op trans = params.trans();
    int64_t n = params.dim.n();
    int64_t kl = params.kl();
    int64_t ku = params.ku();
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
    int64_t kd = 2*kl + ku + 1;  // number of diagonals in factor
    int64_t ldab = roundup( kd, align );
    int64_t ldb = roundup( blas::max( 1, n ), align );
    size_t size_AB = (size_t) ldab * n;
    size_t size_ipiv = (size_t) (n);
    size_t size_B = (size_t) ldb * nrhs;

    std::vector< scalar_t > AB_tst( size_AB );
    std::vector< scalar_t > AB_ref( size_AB );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB_tst.size(), &AB_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    AB_ref = AB_tst;
    B_ref = B_tst;

    if (verbose >= 1) {
        printf( "\n"
                "AB n=%5lld, kl=%5lld, ku=%5lld, kd=%5lld, ldab=%5lld\n"
                "B n=%5lld, nrhs=%5lld, ldb=%5lld\n",
                llong( n ), llong( kl ), llong( ku ), llong( kd ), llong( ldab ),
                llong( n ), llong( nrhs ), llong( ldb ) );
    }
    if (verbose >= 2) {
        printf( "Input data in rows 0 to kl-1 are ignored.\n" );
        printf( "AB = " ); print_matrix( kd, n, &AB_tst[0], ldab );
        printf( "B = " ); print_matrix( n, nrhs, &B_tst[0], ldb );
    }

    // factor
    int64_t info = lapack::gbtrf( n, n, kl, ku, &AB_tst[0], ldab, &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::gbtrf returned error %lld\n", llong( info ) );
    }

    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::gbtrs( trans, n, kl, ku, nrhs, &AB_tst[0], ldab, &ipiv_tst[0], &B_tst[0], ldb );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gbtrs returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::gbtrs( trans, n, kl, ku, nrhs );
    //params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "A_factor = " ); print_matrix( kd, n, &AB_tst[0], ldab );
        printf( "X = " ); print_matrix( n, nrhs, &B_tst[0], ldb );
    }

    if (params.check() == 'y') {
        // ---------- check error
        // Relative backwards error = ||b - Ax|| / (n * ||A|| * ||x||).
        // No gbmm, so loop over RHS.
        // AB_ref rows 0:kl-1 are ignored; start in row kl.
        for (int64_t j = 0; j < nrhs; ++j) {
            // B_ref -= A * B_tst
            cblas_gbmv( CblasColMajor, cblas_trans_const(trans), n, n, kl, ku,
                        -1.0, &AB_ref[ kl ], ldab,
                              &B_tst[ j*ldb ], 1,
                         1.0, &B_ref[ j*ldb ], 1 );
        }
        if (verbose >= 2) {
            printf( "R = " ); print_matrix( n, nrhs, &B_ref[0], ldb );
        }

        real_t error = lapack::lange( lapack::Norm::One, n, nrhs, &B_ref[0], ldb );
        real_t Xnorm = lapack::lange( lapack::Norm::One, n, nrhs, &B_tst[0], ldb );
        real_t Anorm = lapack::langb( lapack::Norm::One, n, kl, ku, &AB_ref[ kl ], ldab );
        error /= (n * Anorm * Xnorm);
        params.error() = error;
        params.okay() = (error < tol);
    }

    if (params.ref() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_gbtrs( op2char(trans), n, kl, ku, nrhs, &AB_tst[0], ldab, &ipiv_ref[0], &B_ref[0], ldb );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gbtrs returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_gbtrs( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gbtrs_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gbtrs_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gbtrs_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gbtrs_work< std::complex<double> >( params, run );
            break;
    }
}
