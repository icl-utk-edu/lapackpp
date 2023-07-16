// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "lapack.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

//------------------------------------------------------------------------------
template< typename scalar_t >
void test_tgsen_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;
    using complex_t = blas::complex_type< scalar_t >;

    // Constants
    const real_t eps = std::numeric_limits<real_t>::epsilon();

    // get & mark input values
    int64_t ijob = params.ijob();
    lapack::Job jobvl = params.jobvl();
    lapack::Job jobvr = params.jobvr();
    bool wantq = jobvl == lapack::Job::Vec;
    bool wantz = jobvr == lapack::Job::Vec;
    int64_t n = params.dim.n();
    int64_t align = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.ref_time();

    if (! run)
        return;

    //---------- setup
    int64_t lda = roundup( blas::max( 1, n ), align );
    int64_t ldb = roundup( blas::max( 1, n ), align );
    int64_t ldq = roundup( blas::max( 1, n ), align );
    int64_t ldz = roundup( blas::max( 1, n ), align );
    int64_t    sdim_tst = -1;
    lapack_int sdim_ref = -1;
    real_t pl_tst = -1;
    real_t pl_ref = -1;
    real_t pr_tst = -1;
    real_t pr_ref = -1;
    size_t size_select = (size_t) n;
    size_t size_alpha  = (size_t) n;
    size_t size_beta   = (size_t) n;
    size_t size_A = (size_t) lda * n;
    size_t size_B = (size_t) ldb * n;
    size_t size_Q = (size_t) ldq * n;
    size_t size_Z = (size_t) ldz * n;
    size_t size_dif = (size_t) 2;

    std::vector< lapack_logical > select( size_select );
    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< complex_t > alpha_tst( size_alpha );
    std::vector< scalar_t > beta_tst( size_beta );
    std::vector< scalar_t > Q_tst( size_Q );
    std::vector< scalar_t > Z_tst( size_Z );
    std::vector< real_t > dif_tst( size_dif );
    std::vector< real_t > dif_ref( size_dif );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    lapack::larnv( idist, iseed, Q_tst.size(), &Q_tst[0] );
    lapack::larnv( idist, iseed, Z_tst.size(), &Z_tst[0] );

    // Factor (A, B) matrices.
    int64_t info_tst = lapack::gges(
        jobvl, jobvr, lapack::Sort::NotSorted, nullptr, n,
        &A_tst[0], lda, &B_tst[0], ldb,
        &sdim_tst, &alpha_tst[0], &beta_tst[0],
        &Q_tst[0], ldq, &Z_tst[0], ldz );
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gges returned error %lld\n", llong( info_tst ) );
        throw blas::Error();
    }

    std::vector< complex_t > alpha_ref = alpha_tst;
    std::vector< scalar_t > beta_ref  = beta_tst;
    std::vector< scalar_t > A_ref = A_tst;
    std::vector< scalar_t > B_ref = B_tst;
    std::vector< scalar_t > Q_ref = Q_tst;
    std::vector< scalar_t > Z_ref = Z_tst;

    // Randomly select ~ 5% of eigenvalues to reorder.
    for (int i = 0; i < n; ++i ) {
        select[ i ] = (rand() / double(RAND_MAX)) < 0.05;
    }

    //---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();

    info_tst = lapack::tgsen(
        ijob, wantq, wantz, &select[0], n,
        &A_tst[0], lda, &B_tst[0], ldb, &alpha_tst[0], &beta_tst[0],
        &Q_tst[0], ldq, &Z_tst[0], ldz,
        &sdim_tst, &pl_tst, &pr_tst, &dif_tst[0] );

    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::tgsen returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;

    if (params.ref() == 'y' || params.check() == 'y') {
        //---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();

        int64_t info_ref = LAPACKE_tgsen(
            ijob, wantq, wantz, &select[0], n,
            &A_ref[0], lda, &B_ref[0], ldb, &alpha_ref[0], &beta_ref[0],
            &Q_ref[0], ldq, &Z_ref[0], ldz,
            &sdim_ref, &pl_ref, &pr_ref, &dif_ref[0] );

        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_tgsen returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;

        //---------- check error compared to reference
        real_t info_error  = (info_tst != 0 || info_ref != 0);
        real_t A_error     = abs_error( A_tst, A_ref );
        real_t B_error     = abs_error( B_tst, B_ref );
        real_t Q_error     = abs_error( Q_tst, Q_ref );
        real_t Z_error     = abs_error( Z_tst, Z_ref );
        real_t alpha_error = abs_error( alpha_tst, alpha_ref );
        real_t beta_error  = abs_error( beta_tst, beta_ref );
        real_t sdim_error  = std::abs( sdim_tst - sdim_ref );
        real_t pl_error    = std::abs( pl_tst - pl_ref );
        real_t pr_error    = std::abs( pr_tst - pr_ref );
        real_t dif_error   = abs_error( dif_tst, dif_ref );
        real_t error = info_error + A_error + B_error + Q_error + Z_error
                     + alpha_error + beta_error + sdim_error
                     + pl_error + pr_error + dif_error;
        params.error() = error;
        params.okay() = (error < eps);  // expect lapackpp ~= lapacke

        if (verbose && error != 0) {
            printf( "error %.3g = info %.3g + A %.3g + B %.3g + Q %.3g + Z %.3g"
                    " + alpha %.3g + beta %.3g + sdim %.3g"
                    " + pl %.3g + pr %.3g + dif %.3g\n",
                    error, info_error, A_error, B_error, Q_error, Z_error,
                    alpha_error, beta_error, sdim_error,
                    pl_error, pr_error, dif_error );
        }
    }
}

//------------------------------------------------------------------------------
void test_tgsen( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_tgsen_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_tgsen_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_tgsen_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_tgsen_work< std::complex<double> >( params, run );
            break;
    }
}
