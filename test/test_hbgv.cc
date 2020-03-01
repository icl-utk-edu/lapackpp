// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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
void test_hbgv_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Job jobz = params.jobz();
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t ka = params.ka();
    int64_t kb = params.kb();
    int64_t align = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();

    if (! run)
        return;

    // skip invalid sizes
    if (! (n >= ka && ka >= kb)) {
        params.msg() = "skipping: requires n >= ka >= kb (not documented)";
        return;
    }

    // ---------- setup
    int64_t ldab = roundup( ka+1, align );
    int64_t ldbb = roundup( kb+1, align );
    int64_t ldz = ( jobz == lapack::Job::Vec ? roundup( blas::max( 1, n ), align ) : 1 );
    size_t size_AB = (size_t) ldab * n;
    size_t size_BB = (size_t) ldbb * n;
    size_t size_W = (size_t) (n);
    size_t size_Z = (size_t) ldz * n;

    std::vector< scalar_t > AB_tst( size_AB );
    std::vector< scalar_t > AB_ref( size_AB );
    std::vector< scalar_t > BB_tst( size_BB );
    std::vector< scalar_t > BB_ref( size_BB );
    std::vector< real_t > W_tst( size_W );
    std::vector< real_t > W_ref( size_W );
    std::vector< scalar_t > Z_tst( size_Z );
    std::vector< scalar_t > Z_ref( size_Z );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB_tst.size(), &AB_tst[0] );
    lapack::larnv( idist, iseed, BB_tst.size(), &BB_tst[0] );

    // diagonally dominant -> positive definite
    if (uplo == lapack::Uplo::Upper) {
        for (int64_t j = 0; j < n; ++j) {
            BB_tst[ kb + j*ldbb ] += n;
        }
    }
    else { // lower
       for (int64_t j = 0; j < n; ++j) {
           BB_tst[ j*ldbb ] += n;
       }
    }
    if (verbose >= 2) {
        printf( "Aband" ); print_matrix( ka+1, n, &AB_tst[0], ldab );
        printf( "Bband" ); print_matrix( kb+1, n, &BB_tst[0], ldab );
    }

    AB_ref = AB_tst;
    BB_ref = BB_tst;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::hbgv( jobz, uplo, n, ka, kb, &AB_tst[0], ldab, &BB_tst[0], ldbb, &W_tst[0], &Z_tst[0], ldz );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::hbgv returned error %lld\n", (lld) info_tst );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::hbgv( jobz, n, ka, kb );
    // params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "W_tst" ); print_vector( n, &W_tst[0], 1 );
        if (jobz == lapack::Job::Vec) {
            printf( "Z_tst" ); print_matrix( n, n, &Z_tst[0], ldz );
        }
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_hbgv( job2char(jobz), uplo2char(uplo), n, ka, kb, &AB_ref[0], ldab, &BB_ref[0], ldbb, &W_ref[0], &Z_ref[0], ldz );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_hbgv returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "W_ref" ); print_vector( n, &W_ref[0], 1 );
            if (jobz == lapack::Job::Vec) {
                printf( "Z_ref" ); print_matrix( n, n, &Z_ref[0], ldz );
            }
        }

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( AB_tst, AB_ref );
        error += abs_error( BB_tst, BB_ref );
        error += abs_error( W_tst, W_ref );
        error += abs_error( Z_tst, Z_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_hbgv( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_hbgv_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_hbgv_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_hbgv_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_hbgv_work< std::complex<double> >( params, run );
            break;
    }
}
