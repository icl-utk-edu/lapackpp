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
void test_tgexc_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;
    using complex_t = blas::complex_type< scalar_t >;

    // Constants
    const real_t eps = std::numeric_limits<real_t>::epsilon();

    // get & mark input values
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
    // Arbitrarily, move row 1/3n to 2/3n.
    int64_t ifst_tst = blas::max( 1, n * 0.33 );
    int64_t ilst_tst = blas::max( 1, n * 0.66 );
    lapack_int ifst_ref = ifst_tst;
    lapack_int ilst_ref = ilst_tst;
    size_t size_A = (size_t) lda * n;
    size_t size_B = (size_t) ldb * n;
    size_t size_Q = (size_t) ldq * n;
    size_t size_Z = (size_t) ldz * n;

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > Q_tst( size_Q );
    std::vector< scalar_t > Z_tst( size_Z );
    std::vector< complex_t > alpha( n );
    std::vector< scalar_t > beta( n );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    lapack::larnv( idist, iseed, Q_tst.size(), &Q_tst[0] );
    lapack::larnv( idist, iseed, Z_tst.size(), &Z_tst[0] );

    // Factor (A, B) matrices.
    int64_t sdim_tst = 0;
    int64_t info_tst = lapack::gges(
        jobvl, jobvr, lapack::Sort::NotSorted, nullptr, n,
        &A_tst[0], lda, &B_tst[0], ldb,
        &sdim_tst, &alpha[0], &beta[0],
        &Q_tst[0], ldq, &Z_tst[0], ldz );
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gges returned error %lld\n", llong( info_tst ) );
        throw blas::Error();
    }

    std::vector< scalar_t > A_ref = A_tst;
    std::vector< scalar_t > B_ref = B_tst;
    std::vector< scalar_t > Q_ref = Q_tst;
    std::vector< scalar_t > Z_ref = Z_tst;

    //---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();

    info_tst = lapack::tgexc(
        wantq, wantz, n,
        &A_tst[0], lda, &B_tst[0], ldb,
        &Q_tst[0], ldq, &Z_tst[0], ldz,
        &ifst_tst, &ilst_tst );

    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::tgexc returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;

    if (params.ref() == 'y' || params.check() == 'y') {
        //---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();

        int64_t info_ref = LAPACKE_tgexc(
            wantq, wantz, n,
            &A_ref[0], lda, &B_ref[0], ldb,
            &Q_ref[0], ldq, &Z_ref[0], ldz,
            &ifst_ref, &ilst_ref );

        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_tgexc returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;

        // LAPACKE has ifst, ilst as [in] instead of [in,out].
        if (ifst_tst != ifst_ref || ilst_tst != ilst_ref) {
            printf( "Note: ifst (%lld & %lld) or ilst (%lld & %lld) differ"
                    " between (LAPACK++ & LAPACKE) due to LAPACKE issue.\n",
                    llong( ifst_tst ), llong( ifst_ref ),
                    llong( ilst_tst ), llong( ilst_ref ) );
        }

        //---------- check error compared to reference
        real_t info_error  = (info_tst != 0 || info_ref != 0);
        real_t A_error     = abs_error( A_tst, A_ref );
        real_t B_error     = abs_error( B_tst, B_ref );
        real_t Q_error     = abs_error( Q_tst, Q_ref );
        real_t Z_error     = abs_error( Z_tst, Z_ref );
        // real_t ifst_error = std::abs( ifst_tst - ifst_ref );  // see above
        // real_t ilst_error = std::abs( ilst_tst - ilst_ref );  // see above
        real_t error = info_error + A_error + B_error + Q_error + Z_error;
        params.error() = error;
        params.okay() = (error < eps);  // expect lapackpp ~= lapacke

        if (verbose && error != 0) {
            printf( "error %.3g = info %.3g + A %.3g + B %.3g + Q %.3g + Z %.3g\n",
                    error, info_error, A_error, B_error, Q_error, Z_error );
        }
    }
}

//------------------------------------------------------------------------------
void test_tgexc( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_tgexc_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_tgexc_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_tgexc_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_tgexc_work< std::complex<double> >( params, run );
            break;
    }
}
