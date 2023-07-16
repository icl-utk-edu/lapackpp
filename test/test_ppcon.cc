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
void test_ppcon_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();

    real_t eps = std::numeric_limits< real_t >::epsilon();

    // mark non-standard output values
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();

    if (! run)
        return;

    // ---------- setup
    real_t anorm;
    real_t rcond_tst;
    real_t rcond_ref;
    size_t size_AP = (size_t) (n*(n+1)/2);

    std::vector< scalar_t > AP( size_AP );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AP.size(), &AP[0] );

    // diagonally dominant -> positive definite
    if (uplo == lapack::Uplo::Upper) {
        for (int64_t i = 0; i < n; ++i) {
            AP[ i + 0.5*(i+1)*i ] += n;
        }
    }
    else { // lower
        for (int64_t i = 0; i < n; ++i) {
            AP[ i + n*i - 0.5*i*(i+1) ] += n;
        }
    }

    anorm = lapack::lansp( lapack::Norm::One, uplo, n, &AP[0] );

    // factor A into LL^T
    int64_t info = lapack::pptrf( uplo, n, &AP[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::pptrf returned error %lld\n", llong( info ) );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::ppcon( uplo, n, &AP[0], anorm, &rcond_tst );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::ppcon returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::ppcon( n );
    //params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_ppcon( uplo2char(uplo), n, &AP[0], anorm, &rcond_ref );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_ppcon returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += std::abs( rcond_tst - rcond_ref );
        params.error() = error;
        params.okay() = (error < 3*eps);
    }
}

// -----------------------------------------------------------------------------
void test_ppcon( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_ppcon_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_ppcon_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_ppcon_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_ppcon_work< std::complex<double> >( params, run );
            break;
    }
}
