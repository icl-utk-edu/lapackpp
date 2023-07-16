// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// this is similar to blaspp/test/test_syr.hh,
// except it uses LAPACK++ instead of calling Fortran LAPACK,
// and tests syr< complex >.


#include "test.hh"
#include "lapack.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "check_gemm2.hh"  // uses LAPACK++ instead of Fortran lapack
#include "cblas_wrappers.hh"
#include "blas/syr.hh"  // from BLAS++

// -----------------------------------------------------------------------------
template< typename TA, typename TX >
void test_syr_work( Params& params, bool run )
{
    using blas::real;
    using blas::imag;
    using scalar_t = blas::scalar_type< TA, TX >;
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Uplo uplo = params.uplo();
    scalar_t alpha  = params.alpha();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t align   = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();
    params.gflops();

    if (! run)
        return;

    // setup
    int64_t lda = roundup( n, align );
    size_t size_A = size_t(lda)*n;
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    std::vector<TA> A   ( size_A );
    std::vector<TA> Aref( size_A );
    std::vector<TX> x   ( size_x );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 0, 0, 1 };
    lapack::generate_matrix( params.matrix, n, n, &A[0], lda );
    lapack::larnv( idist, iseed, size_x, &x[0] );
    Aref = A;

    // norms for error check
    real_t Anorm = lapack::lansy( lapack::Norm::Fro, uplo, n, &A[0], lda );
    real_t Xnorm = blas::nrm2( n, &x[0], std::abs(incx) );

    // test error exits
    if (params.error_exit() == 'y') {
        using blas::Layout;
        using blas::Uplo;
        assert_throw( blas::syr( Layout(0), uplo,     n, alpha, &x[0], incx, &A[0], lda ), blas::Error );
        assert_throw( blas::syr( layout,    Uplo(0),  n, alpha, &x[0], incx, &A[0], lda ), blas::Error );
        assert_throw( blas::syr( layout,    uplo,    -1, alpha, &x[0], incx, &A[0], lda ), blas::Error );
        assert_throw( blas::syr( layout,    uplo,     n, alpha, &x[0],    0, &A[0], lda ), blas::Error );
        assert_throw( blas::syr( layout,    uplo,     n, alpha, &x[0], incx, &A[0], n-1 ), blas::Error );
    }

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n",
                llong( n ), llong( lda ),  llong( size_A ), Anorm,
                llong( n ), llong( incx ), llong( size_x ), Xnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei;\n",
                real(alpha), imag(alpha) );
        printf( "A = " ); print_matrix( n, n, &A[0], lda );
        printf( "x = " ); print_vector( n, &x[0], incx );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    blas::syr( layout, uplo, n, alpha, &x[0], incx, &A[0], lda );
    time = testsweeper::get_wtime() - time;

    params.time() = time * 1000;  // msec
    double gflop = blas::Gflop< scalar_t >::syr( n );
    params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "A2 = " ); print_matrix( n, n, &A[0], lda );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        cblas_syr( cblas_layout_const(layout), cblas_uplo_const(uplo),
                   n, alpha, &x[0], incx, &Aref[0], lda );
        time = testsweeper::get_wtime() - time;

        params.ref_time() = time * 1000;  // msec
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "Aref = " ); print_matrix( n, n, &Aref[0], lda );
        }

        // check error compared to reference
        // beta = 1
        real_t error;
        int64_t okay;
        check_herk( uplo, n, 1, alpha, scalar_t(1), Xnorm, Xnorm, Anorm,
                    &Aref[0], lda, &A[0], lda, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }
}

// -----------------------------------------------------------------------------
void test_syr( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            //test_syr_work< int64_t >( params, run );
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_syr_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_syr_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_syr_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_syr_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;
    }
}
