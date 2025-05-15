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
void test_lantb_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Norm norm = params.norm();
    lapack::Uplo uplo = params.uplo();
    lapack::Diag diag = params.diag();
    int64_t n = params.dim.n();
    int64_t k = params.kd();
    int64_t align = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( k+1, align );
    size_t size_AB = (size_t) ldab * n;

    std::vector< scalar_t > AB( size_AB );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB.size(), &AB[0] );

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    real_t norm_tst = lapack::lantb( norm, uplo, diag, n, k, &AB[0], ldab );
    time = testsweeper::get_wtime() - time;

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::lantb( norm, diag, n, k );
    // params.gflops() = gflop / time;

    if (verbose >= 1) {
        printf( "norm_tst = %.8e\n", norm_tst );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        real_t norm_ref = LAPACKE_lantb( to_char( norm ), to_char( uplo ), to_char( diag ), n, k, &AB[0], ldab );
        time = testsweeper::get_wtime() - time;

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        if (verbose >= 1) {
            printf( "norm_ref = %.8e\n", norm_ref );
        }

        // ---------- check error compared to reference
        // todo: adjust normalize for band
        real_t tol = 3 * std::numeric_limits< real_t >::epsilon();
        real_t normalize = 1;
        if (norm == lapack::Norm::Max && ! blas::is_complex_v< scalar_t >) {
            // max-norm depends on only one element, so in real there should be
            // zero error, but in complex there's error in abs().
            tol = 0;
        }
        else if (norm == lapack::Norm::One)
            normalize = sqrt( real_t(n) );
        else if (norm == lapack::Norm::Inf)
            normalize = sqrt( real_t(n) );
        else if (norm == lapack::Norm::Fro)
            normalize = sqrt( real_t(n)*n );
        real_t error = std::abs( norm_tst - norm_ref ) / normalize;
        if (norm_ref != 0)
            error /= norm_ref;
        params.error() = error;
        params.okay() = (error <= tol);
    }
}

// -----------------------------------------------------------------------------
void test_lantb( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_lantb_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_lantb_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_lantb_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_lantb_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
