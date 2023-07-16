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
#include <iostream>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_ggev_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Job jobvl = params.jobvl();
    lapack::Job jobvr = params.jobvr();
    int64_t n = params.dim.n();
    int64_t align = params.align();
    params.matrix.mark();
    params.matrixB.mark();

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, n ), align );
    int64_t ldb = roundup( blas::max( 1, n ), align );
    int64_t ldvl = ( jobvl == lapack::Job::Vec ? roundup( blas::max(1, n), align ) : 1 );
    int64_t ldvr  = ( jobvr == lapack::Job::Vec ? roundup( blas::max(1, n), align ) : 1 );
    size_t size_A = (size_t)( lda * n );
    size_t size_B = (size_t)( ldb * n );
    size_t size_alpha = (size_t)( n );
    size_t size_beta = (size_t)( n );
    size_t size_VL = (size_t)( ldvl * n );
    size_t size_VR = (size_t)( ldvr * n );

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );
    std::vector< std::complex<real_t> > alpha_tst( size_alpha );
    std::vector< std::complex<real_t> > alpha_ref( size_alpha );
    std::vector< scalar_t > beta_tst( size_beta );
    std::vector< scalar_t > beta_ref( size_beta );
    std::vector< scalar_t > VL_tst( size_VL );
    std::vector< scalar_t > VL_ref( size_VL );
    std::vector< scalar_t > VR_tst( size_VR );
    std::vector< scalar_t > VR_ref( size_VR );

    lapack::generate_matrix( params.matrix,  n, n, &A_tst[0], lda );
    lapack::generate_matrix( params.matrixB, n, n, &B_tst[0], ldb );
    A_ref = A_tst;
    B_ref = B_tst;

    std::copy( alpha_tst.begin(), alpha_tst.end(), alpha_ref.begin() );

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::ggev( jobvl, jobvr, n, &A_tst[0], lda, &B_tst[0], ldb, &alpha_tst[0], &beta_tst[0], &VL_tst[0], ldvl, &VR_tst[0], ldvr );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::ggev returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::ggev( jobvl, jobvr, n );
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_ggev( job2char(jobvl), job2char(jobvr), n, &A_ref[0], lda, &B_ref[0], ldb, &alpha_ref[0], &beta_ref[0], &VL_ref[0], ldvl, &VR_ref[0], ldvr );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_ggev returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( B_tst, B_ref );
        error += abs_error( alpha_tst, alpha_ref );
        error += abs_error( beta_tst, beta_ref );
        error += abs_error( VL_tst, VL_ref );
        error += abs_error( VR_tst, VR_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_ggev( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_ggev_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_ggev_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_ggev_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_ggev_work< std::complex<double> >( params, run );
            break;
    }
}
