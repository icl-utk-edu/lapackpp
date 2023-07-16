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
void test_gtcon_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Norm norm = params.norm();
    int64_t n = params.dim.n();

    // mark non-standard output values
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();

    if (! run)
        return;

    // ---------- setup
    real_t anorm;  // todo value
    real_t rcond_tst;
    real_t rcond_ref;
    size_t size_DL = (size_t) (n-1);
    size_t size_D = (size_t) (n);
    size_t size_DU = (size_t) (n-1);
    size_t size_DU2 = (size_t) (n-2);
    size_t size_ipiv = (size_t) (n);

    std::vector< scalar_t > DL( size_DL );
    std::vector< scalar_t > D( size_D );
    std::vector< scalar_t > DU( size_DU );
    std::vector< scalar_t > DU2( size_DU2 );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, DL.size(), &DL[0] );
    lapack::larnv( idist, iseed, D.size(), &D[0] );
    lapack::larnv( idist, iseed, DU.size(), &DU[0] );
    lapack::larnv( idist, iseed, DU2.size(), &DU2[0] );
    // todo: initialize ipiv_tst and ipiv_ref

    // compute norm
    anorm = lapack::langt( norm, n, &DL[0], &D[0], &DU[0] );

    // factor
    int64_t info = lapack::gttrf( n, &DL[0], &D[0], &DU[0], &DU2[0], &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::gttrf returned error %lld\n", llong( info ) );
    }

    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::gtcon( norm, n, &DL[0], &D[0], &DU[0], &DU2[0], &ipiv_tst[0], anorm, &rcond_tst );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gtcon returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::gtcon( norm, n );
    //params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_gtcon( norm2char(norm), n, &DL[0], &D[0], &DU[0], &DU2[0], &ipiv_ref[0], anorm, &rcond_ref );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gtcon returned error %lld\n", llong( info_ref ) );
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
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gtcon( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gtcon_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gtcon_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gtcon_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gtcon_work< std::complex<double> >( params, run );
            break;
    }
}
