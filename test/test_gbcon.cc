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
void test_gbcon_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Norm norm = params.norm();
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
    int64_t ldab = roundup( 2*kl+ku+1, align );
    real_t anorm;
    real_t rcond_tst;
    real_t rcond_ref;
    size_t size_AB = (size_t) ldab * n;
    size_t size_ipiv = (size_t) (n);

    std::vector< scalar_t > AB( size_AB );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB.size(), &AB[0] );


    anorm = lapack::langb( norm, n, kl, ku, &AB[kl], ldab );

    int64_t info = lapack::gbtrf( n, n, kl, ku, &AB[0], ldab, &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::gbtrf returned error %lld\n", llong( info ) );
    }

    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::gbcon( norm, n, kl, ku, &AB[0], ldab, &ipiv_tst[0], anorm, &rcond_tst );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gbcon returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::gbcon( norm, n, kl, ku );
    //params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_gbcon( norm2char(norm), n, kl, ku, &AB[0], ldab, &ipiv_ref[0], anorm, &rcond_ref );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gbcon returned error %lld\n", llong( info_ref ) );
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
void test_gbcon( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gbcon_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gbcon_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gbcon_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gbcon_work< std::complex<double> >( params, run );
            break;
    }
}
