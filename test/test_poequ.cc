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
void test_poequ_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t align = params.align();
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
    int64_t lda = roundup( blas::max( 1, n ), align );
    real_t scond_tst = 0;
    real_t scond_ref = 0;
    real_t amax_tst;
    real_t amax_ref;
    size_t size_A = (size_t) lda * n;
    size_t size_S = (size_t) (n);

    std::vector< scalar_t > A( size_A );
    std::vector< real_t > S_tst( size_S );
    std::vector< real_t > S_ref( size_S );

    lapack::generate_matrix( params.matrix, n, n, &A[0], lda );

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::poequ( n, &A[0], lda, &S_tst[0], &scond_tst, &amax_tst );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::poequ returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::poequ( n );
    //params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_poequ( n, &A[0], lda, &S_ref[0], &scond_ref, &amax_ref );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_poequ returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( S_tst, S_ref );
        error += std::abs( scond_tst - scond_ref );
        error += std::abs( amax_tst - amax_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_poequ( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_poequ_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_poequ_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_poequ_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_poequ_work< std::complex<double> >( params, run );
            break;
    }
}
