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
#include "check_geev.hh"

#include <vector>
#include <algorithm>

// -----------------------------------------------------------------------------
// comparison operator for sorting
template< typename T >
bool lessthan( T a, T b )
{
    return (real(a) < real(b)) || (real(a) == real(b) && imag(a) < imag(b));
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_geev_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Job jobvl = params.jobvl();
    lapack::Job jobvr = params.jobvr();
    int64_t n = params.dim.n();
    int64_t align = params.align();
    int64_t verbose = params.verbose();
    params.matrix.mark();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol() * eps;

    // mark non-standard output values
    params.error2();
    params.error3();
    params.error4();
    params.error5();
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();

    params.error .name( "A' Vl - Vl W'" );
    params.error2.name( "Vl(j) norm" );
    params.error3.name( "A Vr - Vr W" );
    params.error4.name( "Vr(j) norm" );
    params.error5.name( "W - Wref" );

    params.error .width( 13 );
    params.error2.width( 10 );
    params.error3.width( 11 );
    params.error4.width( 10 );
    params.error5.width(  8 );

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, n ), align );
    int64_t ldvl = roundup( n, align );
    int64_t ldvr = roundup( n, align );
    size_t size_A = (size_t) lda * n;
    size_t size_W = (size_t) (n);
    size_t size_VL = (size_t) ldvl * n;
    size_t size_VR = (size_t) ldvr * n;

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< std::complex<real_t> > W_tst( size_W );
    std::vector< std::complex<real_t> > W_ref( size_W );
    std::vector< scalar_t > VL_tst( size_VL );
    std::vector< scalar_t > VL_ref( size_VL );
    std::vector< scalar_t > VR_tst( size_VR );
    std::vector< scalar_t > VR_ref( size_VR );

    // Generate test matrix
    lapack::generate_matrix( params.matrix, n, n, &A_tst[0], lda );

    A_ref = A_tst;

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld\n",
                llong( n ), llong( lda ) );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( n, n, &A_tst[0], lda );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    //printf (" test start\n");
    int64_t info_tst = lapack::geev( jobvl, jobvr, n, &A_tst[0], lda, &W_tst[0], &VL_tst[0], ldvl, &VR_tst[0], ldvr );
    //printf (" test done\n");
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::geev returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::geev( jobvl, jobvr, n );
    //params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "W = " ); print_vector( n, &W_tst[0], 1 );
        if (jobvl == lapack::Job::Vec) {
            printf( "VL = " ); print_matrix( n, n, &VL_tst[0], ldvl );
        }
        if (jobvr == lapack::Job::Vec) {
            printf( "VR = " ); print_matrix( n, n, &VR_tst[0], ldvr );
        }
    }

    bool okay = true;
    if (params.check() == 'y') {
        // ---------- check numerical error
        // formula from get22; differs from LAWN 41, usess ||V||_1 instead of n
        // 1. || A^H Vl - Vl W^H ||_1 / (||V||_1 ||A||_1)
        // 2. max_{j=1, ..., n} | || Vl(j) ||_2 - 1 |
        // 3. || A Vr - Vr W || / (||V||_1 ||A||_1)
        // 4. max_{j=1, ..., n} | || Vr(j) ||_2 - 1 |
        real_t results[4] = { real_t( testsweeper::no_data_flag ),
                              real_t( testsweeper::no_data_flag ),
                              real_t( testsweeper::no_data_flag ),
                              real_t( testsweeper::no_data_flag ) };
        if (jobvl == lapack::Job::Vec) {
            check_geev( blas::Op::ConjTrans, n, &A_ref[0], lda, &W_tst[0],
                        &VL_tst[0], ldvl, verbose, &results[0] );
            okay = (okay && results[0] < tol && results[1] < tol);
            params.error () = results[0];
            params.error2() = results[1];
        }
        if (jobvr == lapack::Job::Vec) {
            check_geev( blas::Op::NoTrans, n, &A_ref[0], lda, &W_tst[0],
                        &VR_tst[0], ldvr, verbose, &results[2] );
            okay = (okay && results[2] < tol && results[3] < tol);
            params.error3() = results[2];
            params.error4() = results[3];
        }
    }

    if (params.ref() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
    //printf (" ref start\n");
        int64_t info_ref = LAPACKE_geev( job2char(jobvl), job2char(jobvr), n, &A_ref[0], lda, &W_ref[0], &VL_ref[0], ldvl, &VR_ref[0], ldvr );
    //printf (" ref done\n");
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_geev returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "// note: may be sorted differently than results above\n" );
            printf( "Wref  = " ); print_vector( n, &W_ref[0], 1 );
            if (jobvl == lapack::Job::Vec) {
                printf( "VLref = " ); print_matrix( n, n, &VL_ref[0], ldvl );
            }
            if (jobvr == lapack::Job::Vec) {
                printf( "VRref = " ); print_matrix( n, n, &VR_ref[0], ldvr );
            }
        }

        // ---------- check error compared to reference
        // sort eigenvalues into lexical order for comparison
        real_t error = 0;
        std::sort( W_tst.begin(), W_tst.end(), lessthan< std::complex<real_t> > );
        std::sort( W_ref.begin(), W_ref.end(), lessthan< std::complex<real_t> > );
        error = rel_error( W_tst, W_ref );
        okay = (okay && error < tol);
        params.error5() = error;
    }

    // okay from error ... error5
    params.okay() = okay;
}

// -----------------------------------------------------------------------------
void test_geev( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_geev_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_geev_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_geev_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_geev_work< std::complex<double> >( params, run );
            break;
    }
}
