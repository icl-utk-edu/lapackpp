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
void test_gbequ_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t kl = params.kl();
    int64_t ku = params.ku();
    int64_t align = params.align();

    // mark non-standard output values
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( kl+ku+1, align );
    real_t rowcnd_tst = 0;
    real_t rowcnd_ref = 0;
    real_t colcnd_tst = 0;
    real_t colcnd_ref = 0;
    real_t amax_tst;
    real_t amax_ref;
    size_t size_AB = (size_t) ldab * n;
    size_t size_R = (size_t) (m);
    size_t size_C = (size_t) (n);

    std::vector< scalar_t > AB( size_AB );
    std::vector< real_t > R_tst( size_R );
    std::vector< real_t > R_ref( size_R );
    std::vector< real_t > C_tst( size_C );
    std::vector< real_t > C_ref( size_C );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB.size(), &AB[0] );

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::gbequ( m, n, kl, ku, &AB[0], ldab, &R_tst[0], &C_tst[0], &rowcnd_tst, &colcnd_tst, &amax_tst );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gbequ returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::gbequ( m, n, kl, ku );
    //params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_gbequ( m, n, kl, ku, &AB[0], ldab, &R_ref[0], &C_ref[0], &rowcnd_ref, &colcnd_ref, &amax_ref );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gbequ returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( R_tst, R_ref );
        error += abs_error( C_tst, C_ref );
        error += std::abs( rowcnd_tst - rowcnd_ref );
        error += std::abs( colcnd_tst - colcnd_ref );
        error += std::abs( amax_tst - amax_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gbequ( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gbequ_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gbequ_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gbequ_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gbequ_work< std::complex<double> >( params, run );
            break;
    }
}
