// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "check_gemm2.hh"
#include "cblas_wrappers.hh"

// -----------------------------------------------------------------------------
template< typename TA, typename TX, typename TY >
void test_symv_work( Params& params, bool run )
{
    using namespace testsweeper;
    using namespace blas;
    using blas::real;
    using blas::imag;
    using scalar_t = blas::scalar_type<TA, TX, TY>;
    using real_t = blas::real_type<scalar_t>;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Uplo uplo = params.uplo();
    scalar_t alpha  = params.alpha();
    scalar_t beta   = params.beta();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
    int64_t align   = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.gflops();
    params.gbytes();
    params.ref_time();
    params.ref_gflops();
    params.ref_gbytes();

    // adjust header to msec
    params.time.name( "time (ms)" );
    params.ref_time.name( "time (ms)" );

    if (! run)
        return;

    // setup
    int64_t lda = roundup( n, align );
    size_t size_A = size_t(lda)*n;
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    size_t size_y = (n - 1) * std::abs(incy) + 1;
    std::vector<TA> A   ( size_A );
    std::vector<TX> x   ( size_x );
    std::vector<TY> y   ( size_y );
    std::vector<TY> yref( size_y );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::generate_matrix( params.matrix, n, n, &A[0], lda );
    lapack::larnv( idist, iseed, x.size(), &x[0] );
    lapack::larnv( idist, iseed, y.size(), &y[0] );
    yref = y;

    // norms for error check
    real_t Anorm = lapack::lansy( lapack::Norm::Fro, uplo, n, &A[0], lda );
    real_t Xnorm = blas::nrm2( n, &x[0], std::abs(incx) );
    real_t Ynorm = blas::nrm2( n, &y[0], std::abs(incy) );

    // test error exits
    if (params.error_exit() == 'y') {
        using blas::Layout;
        using blas::Uplo;
        assert_throw( blas::symv( Layout(0), uplo,     n, alpha, &A[0], lda, &x[0], incx, beta, &y[0], incy ), blas::Error );
        assert_throw( blas::symv( layout,    Uplo(0),  n, alpha, &A[0], lda, &x[0], incx, beta, &y[0], incy ), blas::Error );
        assert_throw( blas::symv( layout,    uplo,    -1, alpha, &A[0], lda, &x[0], incx, beta, &y[0], incy ), blas::Error );
        assert_throw( blas::symv( layout,    uplo,     n, alpha, &A[0], n-1, &x[0], incx, beta, &y[0], incy ), blas::Error );
        assert_throw( blas::symv( layout,    uplo,     n, alpha, &A[0], lda, &x[0],    0, beta, &y[0], incy ), blas::Error );
        assert_throw( blas::symv( layout,    uplo,     n, alpha, &A[0], lda, &x[0], incx, beta, &y[0],    0 ), blas::Error );
    }

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n"
                "y n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n",
                llong( n ), llong( lda ),  llong( size_A ), Anorm,
                llong( n ), llong( incx ), llong( size_x ), Xnorm,
                llong( n ), llong( incy ), llong( size_y ), Ynorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( n, n, &A[0], lda );
        printf( "x    = " ); print_vector( n, &x[0], incx );
        printf( "y    = " ); print_vector( n, &y[0], incy );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::symv( layout, uplo, n, alpha, &A[0], lda, &x[0], incx, beta, &y[0], incy );
    time = get_wtime() - time;

    double gflop = Gflop < scalar_t >::symv( n );
    double gbyte = Gbyte < scalar_t >::symv( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    if (verbose >= 2) {
        printf( "y2   = " ); print_vector( n, &y[0], incy );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_symv( cblas_layout_const(layout), cblas_uplo_const(uplo), n,
                    alpha, &A[0], lda, &x[0], incx, beta, &yref[0], incy );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 2) {
            printf( "yref = " ); print_vector( n, &yref[0], incy );
        }

        // check error compared to reference
        // treat y as 1 x leny matrix with ld = incy; k = lenx is reduction dimension
        real_t error;
        int64_t okay;
        check_gemm( 1, n, n,
                alpha, beta,
                Anorm, Xnorm, Ynorm,
                &yref[0], std::abs(incy),
                &y[0], std::abs(incy),
                &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }
}

// -----------------------------------------------------------------------------
void test_symv( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();

        case testsweeper::DataType::Single:
            test_symv_work< float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_symv_work< double, double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_symv_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_symv_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;
    }
}
