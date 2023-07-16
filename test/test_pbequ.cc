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
void test_pbequ_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t kd = params.kd();
    int64_t align = params.align();

    // mark non-standard output values
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( kd+1, align );
    real_t scond_tst = 0;
    real_t scond_ref = 0;
    real_t amax_tst;
    real_t amax_ref;
    size_t size_AB = (size_t) ldab * n;
    size_t size_S = (size_t) (n);

    std::vector< scalar_t > AB( size_AB );
    std::vector< real_t > S_tst( size_S );
    std::vector< real_t > S_ref( size_S );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB.size(), &AB[0] );

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

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::pbequ( uplo, n, kd, &AB[0], ldab, &S_tst[0], &scond_tst, &amax_tst );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::pbequ returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::pbequ( n, kd );
    //params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_pbequ( uplo2char(uplo), n, kd, &AB[0], ldab, &S_ref[0], &scond_ref, &amax_ref );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_pbequ returned error %lld\n", llong( info_ref ) );
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
void test_pbequ( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_pbequ_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_pbequ_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_pbequ_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_pbequ_work< std::complex<double> >( params, run );
            break;
    }
}
