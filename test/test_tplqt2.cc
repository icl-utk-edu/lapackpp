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

#if LAPACK_VERSION >= 30700  // >= 3.7.0

// -----------------------------------------------------------------------------
// tplqt is blocked version
// tplqt2 is non-blocked version
// todo: merge test_tplqt2.cc and test_tplqt.cc
template< typename scalar_t >
void test_tplqt2_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t L = params.l();
    int64_t align = params.align();
    int64_t verbose = params.verbose();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol() * eps;

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();
    params.gflops();
    params.msg();

    if (! run)
        return;

    // skip invalid sizes
    if (blas::min(m, n) < L) {
        params.msg() = "skipping: requires min(m, n) >= L >= 0";
        return;
    }

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, m ), align );
    int64_t ldb = roundup( blas::max( 1, m ), align );
    int64_t ldt = roundup( blas::max( 1, m ), align );
    size_t size_A = (size_t) lda * m;  // m-by-m
    size_t size_B = (size_t) ldb * n;  // m-by-n
    size_t size_T = (size_t) ldt * m;  // m-by-m

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );
    std::vector< scalar_t > T_tst( size_T );
    std::vector< scalar_t > T_ref( size_T );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    A_ref = A_tst;
    B_ref = B_tst;

    if (verbose >= 1) {
        printf( "\n"
                "A m-by-m=%5lld, lda=%5lld triangle\n"
                "B m=%5lld, n=%5lld, ldb=%5lld, rows m-L=%5lld rect, L=%5lld trapezoid\n",
                llong( m ), llong( lda ),
                llong( m ), llong( n ), llong( ldb ), llong( (m-L) ), llong( L ) );
    }
    if (verbose >= 2) {
        printf( "A  = " ); print_matrix( m, m,   &A_tst[0], lda );
        printf( "B1 = " ); print_matrix( m, n-L, &B_tst[0], ldb );
        printf( "B2 = " ); print_matrix( m, L,   &B_tst[(n-L)*ldb], ldb );
        printf( "\n" );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::tplqt2(
        m, n, L, &A_tst[0], lda, &B_tst[0], ldb, &T_tst[0], ldt );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::tplqt2 returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::gelqf( m, n );  // under-estimate
    params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "Ahat  = " ); print_matrix( m,  m,   &A_tst[0], lda );
        printf( "Bhat1 = " ); print_matrix( m,  n-L, &B_tst[0], ldb );
        printf( "Bhat2 = " ); print_matrix( m,  L,   &B_tst[(n-L)*ldb], ldb );
        printf( "T = "     ); print_matrix( m,  m,   &T_tst[0], ldt );
        printf( "\n" );
    }

    if (params.check() == 'y') {
        // ---------- check error
        // Relative backwards error = ||A - LQ|| / (m * ||A||).
        // todo: Orthogonality check = ||I - Q^H Q|| / m.

        // LA = lower( A ); LB = zeros( m, n );
        std::vector< scalar_t > LA( size_A, 0.0 );
        std::vector< scalar_t > LB( size_B, 0.0 );
        lapack::lacpy( lapack::MatrixType::Lower, m, m, &A_tst[0], lda, &LA[0], lda );
        if (verbose >= 2) {
            printf( "LA  = " ); print_matrix( m, m,   &LA[0], lda );
            printf( "LB1 = " ); print_matrix( m, n-L, &LB[0], ldb );
            printf( "LB2 = " ); print_matrix( m, L,   &LB[(n-L)*ldb], ldb );
            printf( "\n" );
        }

        // zero out upper triangle of A (above col 1) and B (above col n-L+1).
        lapack::laset( lapack::MatrixType::Upper, m-1, m-1, 0.0, 0.0, &A_ref[1*lda], lda );
        lapack::laset( lapack::MatrixType::Upper, L-1, L-1, 0.0, 0.0, &B_ref[(n-L+1)*ldb], ldb );
        if (verbose >= 2) {
            printf( "Aref  = " ); print_matrix( m, m,   &A_ref[0], lda );
            printf( "Bref1 = " ); print_matrix( m, n-L, &B_ref[0], ldb );
            printf( "Bref2 = " ); print_matrix( m, L,   &B_ref[(n-L)*ldb], ldb );
            printf( "\n" );
        }

        // [ LA LB ] = [ LA LB ] * Q
        info_tst = lapack::tpmlqt(
            lapack::Side::Right, lapack::Op::NoTrans, m, n, m, L, m,
            &B_tst[0], ldb, &T_tst[0], ldt, &LA[0], lda, &LB[0], ldb );
        assert( info_tst == 0 );

        real_t Anorm = std::max(
            lapack::lange( lapack::Norm::One, m, m, &A_ref[0], lda ),
            lapack::lange( lapack::Norm::One, m, n, &B_ref[0], ldb ) );

        // [ A B ] - ([ LA LB ] * Q)
        blas::axpy( size_A, -1.0, &LA[0], 1, &A_ref[0], 1 );
        blas::axpy( size_B, -1.0, &LB[0], 1, &B_ref[0], 1 );

        real_t error = std::max(
            lapack::lange( lapack::Norm::One, m, m, &A_ref[0], lda ),
            lapack::lange( lapack::Norm::One, m, n, &B_ref[0], ldb ) );
        error /= ((m + n)*Anorm);
        params.error() = error;
        params.okay() = (error < tol);

        if (verbose >= 2) {
            printf( "A_QR0 = " ); print_matrix( m, m,   &A_ref[0],   lda );
            printf( "A_QR1 = " ); print_matrix( m, n-L, &B_ref[0],   ldb );
            printf( "A_QR2 = " ); print_matrix( m, L,   &B_ref[m-L], ldb );
            printf( "\n" );
        }
    }

    if (params.ref() == 'y') {
        // ---------- run reference
        // Reset A, B in case check wiped them out.
        lapack::larnv( idist, iseed, A_ref.size(), &A_ref[0] );
        lapack::larnv( idist, iseed, B_ref.size(), &B_ref[0] );
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_tplqt2(
            m, n, L, &A_ref[0], lda, &B_ref[0], ldb, &T_ref[0], ldt );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_tplqt2 returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_tplqt2( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_tplqt2_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_tplqt2_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_tplqt2_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_tplqt2_work< std::complex<double> >( params, run );
            break;
    }
}

#else

// -----------------------------------------------------------------------------
void test_tplqt2( Params& params, bool run )
{
    fprintf( stderr, "tplqt2 requires LAPACK >= 3.7.0\n\n" );
    exit(0);
}

#endif  // LAPACK >= 3.7.0
