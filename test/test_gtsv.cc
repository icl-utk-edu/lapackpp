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
void test_gtsv_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
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
    size_t size_DL = (size_t) (n-1);
    size_t size_D = (size_t) (n);
    size_t size_DU = (size_t) (n-1);
    size_t size_B = (size_t) ldb * nrhs;

    std::vector< scalar_t > DL_tst( size_DL );
    std::vector< scalar_t > DL_ref( size_DL );
    std::vector< scalar_t > D_tst( size_D );
    std::vector< scalar_t > D_ref( size_D );
    std::vector< scalar_t > DU_tst( size_DU );
    std::vector< scalar_t > DU_ref( size_DU );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, DL_tst.size(), &DL_tst[0] );
    lapack::larnv( idist, iseed, D_tst.size(), &D_tst[0] );
    lapack::larnv( idist, iseed, DU_tst.size(), &DU_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    DL_ref = DL_tst;
    D_ref = D_tst;
    DU_ref = DU_tst;
    B_ref = B_tst;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::gtsv( n, nrhs, &DL_tst[0], &D_tst[0], &DU_tst[0], &B_tst[0], ldb );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gtsv returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::gtsv( n, nrhs );
    //params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_gtsv( n, nrhs, &DL_ref[0], &D_ref[0], &DU_ref[0], &B_ref[0], ldb );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gtsv returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( DL_tst, DL_ref );
        error += abs_error( D_tst, D_ref );
        error += abs_error( DU_tst, DU_ref );
        error += abs_error( B_tst, B_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gtsv( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gtsv_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gtsv_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gtsv_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gtsv_work< std::complex<double> >( params, run );
            break;
    }
}
