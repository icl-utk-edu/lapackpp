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

#if LAPACK_VERSION >= 30400  // >= 3.4.0

// -----------------------------------------------------------------------------
// tpqrt is blocked version
// tpqrt2 is non-blocked version
// todo: merge test_tpqrt2.cc and test_tpqrt.cc
template< typename scalar_t >
void test_tpqrt_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t L = params.l();
    int64_t nb = params.nb();
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
    if (blas::min(m, n) < L || n < nb || nb < 1) {
        params.msg() = "skipping: requires min(m, n) >= L >= 0 and n >= nb >= 1";
        return;
    }

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, n ), align );
    int64_t ldb = roundup( blas::max( 1, m ), align );
    int64_t ldt = roundup( nb, align );
    size_t size_A = (size_t) lda * n;  // n-by-n
    size_t size_B = (size_t) ldb * n;  // m-by-n
    size_t size_T = (size_t) ldt * n;  // nb-by-n

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
                "A n-by-n=%5lld, lda=%5lld triangle\n"
                "B m=%5lld, n=%5lld, ldb=%5lld, rows m-L=%5lld rect, L=%5lld trapezoid\n",
                llong( n ), llong( lda ),
                llong( m ), llong( n ), llong( ldb ), llong( (m-L) ), llong( L ) );
    }
    if (verbose >= 2) {
        printf( "A  = " ); print_matrix( n,   n, &A_tst[0],   lda );
        printf( "B1 = " ); print_matrix( m-L, n, &B_tst[0],   ldb );
        printf( "B2 = " ); print_matrix( L,   n, &B_tst[m-L], ldb );
        printf( "\n" );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::tpqrt(
        m, n, L, nb, &A_tst[0], lda, &B_tst[0], ldb, &T_tst[0], ldt );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::tpqrt returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::geqrf( m, n );  // under-estimate
    params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "Ahat  = " ); print_matrix( n,   n, &A_tst[0],   lda );
        printf( "Bhat1 = " ); print_matrix( m-L, n, &B_tst[0],   ldb );
        printf( "Bhat2 = " ); print_matrix( L,   n, &B_tst[m-L], ldb );
        printf( "T = "     ); print_matrix( nb,  n, &T_tst[0],   ldt );
        printf( "\n" );
    }

    if (params.check() == 'y') {
        // ---------- check error
        // Relative backwards error = ||A - QR|| / (m * ||A||).
        // todo: Orthogonality check = ||I - Q^H Q|| / m.

        // RA = upper( A ); RB = zeros( m, n );
        std::vector< scalar_t > RA( size_A, 0.0 );
        std::vector< scalar_t > RB( size_B, 0.0 );
        lapack::lacpy( lapack::MatrixType::Upper, n, n, &A_tst[0], lda, &RA[0], lda );
        if (verbose >= 2) {
            printf( "RA  = " ); print_matrix( n,   n, &RA[0],   lda );
            printf( "RB1 = " ); print_matrix( m-L, n, &RB[0],   ldb );
            printf( "RB2 = " ); print_matrix( L,   n, &RB[m-L], ldb );
            printf( "\n" );
        }

        // zero out lower triangle of A (below row 1) and B (below row m-L+1).
        lapack::laset( lapack::MatrixType::Lower, n-1, n-1, 0.0, 0.0, &A_ref[1], lda );
        lapack::laset( lapack::MatrixType::Lower, L-1, L-1, 0.0, 0.0, &B_ref[m-L+1], ldb );
        if (verbose >= 2) {
            printf( "Aref  = " ); print_matrix( n,   n, &A_ref[0],   lda );
            printf( "Bref1 = " ); print_matrix( m-L, n, &B_ref[0],   ldb );
            printf( "Bref2 = " ); print_matrix( L,   n, &B_ref[m-L], ldb );
            printf( "\n" );
        }

        // [ RA ] = Q * [ RA ]
        // [ RB ]       [ RB ]
        info_tst = lapack::tpmqrt(
            lapack::Side::Left, lapack::Op::NoTrans, m, n, n, L, nb,
            &B_tst[0], ldb, &T_tst[0], ldt, &RA[0], lda, &RB[0], ldb );
        assert( info_tst == 0 );

        // This isn't || [A ; B] ||, but is an upper bound.
        real_t Anorm = lapack::lange( lapack::Norm::One, n, n, &A_ref[0], lda )
                     + lapack::lange( lapack::Norm::One, m, n, &B_ref[0], ldb );

        // [ A ] - (Q * [ RA ])
        // [ B ]   (    [ RB ])
        blas::axpy( size_A, -1.0, &RA[0], 1, &A_ref[0], 1 );
        blas::axpy( size_B, -1.0, &RB[0], 1, &B_ref[0], 1 );

        // Again, this isn't || [A ; B] ||, but is an upper bound.
        real_t error = lapack::lange( lapack::Norm::One, n, n, &A_ref[0], lda )
                     + lapack::lange( lapack::Norm::One, m, n, &B_ref[0], ldb );
        error /= ((m + n)*Anorm);
        params.error() = error;
        params.okay() = (error < tol);

        if (verbose >= 2) {
            printf( "A_QR0 = " ); print_matrix( n,   n, &A_ref[0],   lda );
            printf( "A_QR1 = " ); print_matrix( m-L, n, &B_ref[0],   ldb );
            printf( "A_QR2 = " ); print_matrix( L,   n, &B_ref[m-L], ldb );
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
        int64_t info_ref = LAPACKE_tpqrt(
            m, n, L, nb, &A_ref[0], lda, &B_ref[0], ldb, &T_ref[0], ldt );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_tpqrt returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_tpqrt( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_tpqrt_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_tpqrt_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_tpqrt_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_tpqrt_work< std::complex<double> >( params, run );
            break;
    }
}

#else

// -----------------------------------------------------------------------------
void test_tpqrt( Params& params, bool run )
{
    fprintf( stderr, "tpqrt requires LAPACK >= 3.4.0\n\n" );
    exit(0);
}

#endif  // LAPACK >= 3.4.0
