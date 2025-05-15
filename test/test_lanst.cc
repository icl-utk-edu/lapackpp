// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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
void test_lanst_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Norm norm = params.norm();
    int64_t n = params.dim.n();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();
    params.msg();

    if (! run)
        return;

    // ---------- setup
    size_t size_D = (size_t) (n);
    size_t size_E = (size_t) (n-1);

    std::vector< real_t > D( size_D );
    std::vector< scalar_t > E( size_E );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, D.size(), &D[0] );
    lapack::larnv( idist, iseed, E.size(), &E[0] );

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    real_t norm_tst = lapack::lanst( norm, n, &D[0], &E[0] );
    time = testsweeper::get_wtime() - time;

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::lanst( norm, n );
    // params.gflops() = gflop / time;

    if (verbose >= 1) {
        printf( "norm_tst = %.8e\n", norm_tst );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        real_t norm_ref = LAPACKE_lanst( to_char( norm ), n, &D[0], &E[0] );
        time = testsweeper::get_wtime() - time;

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        if (verbose >= 1) {
            printf( "norm_ref = %.8e\n", norm_ref );
        }

        // ---------- check error compared to reference
        real_t tol = 3 * std::numeric_limits< real_t >::epsilon();
        if (norm == lapack::Norm::Max && ! blas::is_complex_v< scalar_t >) {
            // max-norm depends on only one element, so in real there should be
            // zero error, but in complex there's error in abs().
            tol = 0;
        }
        real_t error = std::abs( norm_tst - norm_ref );
        if (norm_ref != 0)
            error /= norm_ref;
        params.error() = error;
        params.okay() = (error <= tol);
    }
}

// -----------------------------------------------------------------------------
void test_lanst( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_lanst_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_lanst_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
        case testsweeper::DataType::DoubleComplex:
            params.msg() = "skipping: no complex version";
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
