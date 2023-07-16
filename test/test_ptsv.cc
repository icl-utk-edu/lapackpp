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
void test_ptsv_work( Params& params, bool run )
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
    size_t size_D = (size_t) (n);
    size_t size_E = (size_t) (n-1);
    size_t size_B = (size_t) ldb * nrhs;

    std::vector< real_t > D_tst( size_D );
    std::vector< real_t > D_ref( size_D );
    std::vector< scalar_t > E_tst( size_E );
    std::vector< scalar_t > E_ref( size_E );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, D_tst.size(), &D_tst[0] );
    lapack::larnv( idist, iseed, E_tst.size(), &E_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    E_ref = E_tst;
    B_ref = B_tst;

    // diagonally dominant -> positive definite
    for (int64_t i = 0; i < n; ++i) {
        D_tst[ i ] += n;
    }
    D_ref = D_tst;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::ptsv( n, nrhs, &D_tst[0], &E_tst[0], &B_tst[0], ldb );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::ptsv returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::ptsv( n, nrhs );
    //params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_ptsv( n, nrhs, &D_ref[0], &E_ref[0], &B_ref[0], ldb );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_ptsv returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( D_tst, D_ref );
        error += abs_error( E_tst, E_ref );
        error += abs_error( B_tst, B_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_ptsv( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_ptsv_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_ptsv_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_ptsv_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_ptsv_work< std::complex<double> >( params, run );
            break;
    }
}
