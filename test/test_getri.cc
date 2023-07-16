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
void test_getri_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t align = params.align();
    int64_t verbose = params.verbose();
    params.matrix.mark();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol() * eps;

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();
    params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, n ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_ipiv = (size_t) (n);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );

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

    // factor A into LU
    int64_t info = lapack::getrf( n, n, &A_tst[0], lda, &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::getrf returned error %lld\n", llong( info ) );
    }

    // test error exits
    if (params.error_exit() == 'y') {
        assert_throw( lapack::getri( -1, &A_tst[0], lda, &ipiv_tst[0] ), lapack::Error );
        assert_throw( lapack::getri(  n, &A_tst[0], n-1, &ipiv_tst[0] ), lapack::Error );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::getri( n, &A_tst[0], lda, &ipiv_tst[0] );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::getri returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::getri( n );
    params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "A2 = " ); print_matrix( n, n, &A_tst[0], lda );
    }

    if (params.check() == 'y') {
        // ---------- check error
        // comparing to ref. solution doesn't work due to roundoff errors
        // R = I
        std::vector< scalar_t > R( size_A );
        // todo: laset; needs uplo=general
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < n; ++i) {
                R[ i + j*lda ] = 0;
            }
            R[ j + j*lda ] = 1;
        }

        // R = I - A A^{-1}
        blas::gemm( blas::Layout::ColMajor,
                    blas::Op::NoTrans, blas::Op::NoTrans, n, n, n,
                    -1.0, &A_ref[0], lda,
                          &A_tst[0], lda,
                     1.0, &R[0], lda );
        if (verbose >= 2) {
            printf( "R = " ); print_matrix( n, n, &R[0], lda );
        }

        // error = ||I - A A^{-1}|| / (n ||A|| ||A^{-1}||)
        real_t Rnorm     = lapack::lange( lapack::Norm::Fro, n, n, &R[0],     lda );
        real_t Anorm     = lapack::lange( lapack::Norm::Fro, n, n, &A_ref[0], lda );
        real_t Ainv_norm = lapack::lange( lapack::Norm::Fro, n, n, &A_tst[0], lda );
        real_t error = Rnorm / (n * Anorm * Ainv_norm);
        params.error() = error;
        params.okay() = (error < tol);
    }

    if (params.ref() == 'y') {
        // factor A into LU
        info = LAPACKE_getrf( n, n, &A_ref[0], lda, &ipiv_ref[0] );
        if (info != 0) {
            fprintf( stderr, "LAPACKE_getrf returned error %lld\n", llong( info ) );
        }

        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_getri( n, &A_ref[0], lda, &ipiv_ref[0] );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_getri returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "A2ref = " ); print_matrix( n, n, &A_ref[0], lda );
        }
    }
}

// -----------------------------------------------------------------------------
void test_getri( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_getri_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_getri_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_getri_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_getri_work< std::complex<double> >( params, run );
            break;
    }
}
