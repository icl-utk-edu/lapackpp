// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "lapack.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_laed4_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t i = params.i();
    scalar_t rho = std::abs( params.alpha() );

    assert( 0 <= i && i < n );  // 0-based

    // mark non-standard output values
    params.ref_time();

    if (! run)
        return;

    // ---------- setup
    scalar_t lambda_tst;
    scalar_t lambda_ref;

    std::vector< scalar_t > d( n );
    std::vector< scalar_t > z( n );
    std::vector< scalar_t > delta_tst( n );
    std::vector< scalar_t > delta_ref( n );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, d.size(), &d[0] );
    lapack::larnv( idist, iseed, z.size(), &z[0] );

    // sort d.
    std::sort( d.begin(), d.end() );

    // z should have unit norm.
    real_t z_norm = blas::nrm2( n, &z[0], 1 );
    for (int64_t i = 0; i < n; ++i)
        z[ i ] /= z_norm;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::laed4( n, i, &d[0], &z[0],
                                      &delta_tst[0], rho, &lambda_tst );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::laed4 returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_laed4( n, i, &d[0], &z[0],
                                          &delta_ref[0], rho, &lambda_ref );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_laed4 returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( delta_tst, delta_ref );
        error += std::abs( lambda_tst - lambda_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_laed4( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_laed4_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_laed4_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
        case testsweeper::DataType::DoubleComplex:
            params.msg() = "skipping: no complex version";
            break;
    }
}
