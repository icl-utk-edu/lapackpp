// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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
void test_tplqt_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t L = params.l();
    int64_t mb = params.nb();  // rename nb => mb
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
    if (blas::min(m, n) < L || m < mb || mb < 1) {
        params.msg() = "skipping: requires min(m, n) >= L >= 0 and m >= mb >= 1";
        return;
    }

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, m ), align );
    int64_t ldb = roundup( blas::max( 1, m ), align );
    int64_t ldt = roundup( mb, align );
    size_t size_A = (size_t) lda * m;  // m-by-m
    size_t size_B = (size_t) ldb * n;  // m-by-n
    size_t size_T = (size_t) ldt * m;  // mb-by-m

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

    print_matrix( "A",  m, m,   &A_tst[0], lda, verbose );
    print_matrix( "B1", m, n-L, &B_tst[0], ldb, verbose );
    print_matrix( "B2", m, L,   &B_tst[(n-L)*ldb], ldb, verbose );

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::tplqt(
        m, n, L, mb, &A_tst[0], lda, &B_tst[0], ldb, &T_tst[0], ldt );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::tplqt returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::gelqf( m, n );  // under-estimate
    params.gflops() = gflop / time;

    print_matrix( "Ahat",  m,  m,   &A_tst[0], lda, verbose );
    print_matrix( "Bhat1", m,  n-L, &B_tst[0], ldb, verbose );
    print_matrix( "Bhat2", m,  L,   &B_tst[(n-L)*ldb], ldb, verbose );
    print_matrix( "T",     mb, m,   &T_tst[0], ldt, verbose );

    if (params.check() == 'y') {
        // ---------- check error
        // Relative backwards error = ||A - LQ|| / (m * ||A||).
        // todo: Orthogonality check = ||I - Q^H Q|| / m.

        // LA = lower( A ); LB = zeros( m, n );
        std::vector< scalar_t > LA( size_A, 0.0 );
        std::vector< scalar_t > LB( size_B, 0.0 );
        lapack::lacpy( lapack::MatrixType::Lower, m, m, &A_tst[0], lda, &LA[0], lda );

        print_matrix( "LA",  m, m,   &LA[0], lda, verbose );
        print_matrix( "LB1", m, n-L, &LB[0], ldb, verbose );
        print_matrix( "LB2", m, L,   &LB[(n-L)*ldb], ldb, verbose );

        // zero out upper triangle of A (above col 1) and B (above col n-L+1).
        lapack::laset( lapack::MatrixType::Upper, m-1, m-1, 0.0, 0.0, &A_ref[1*lda], lda );
        lapack::laset( lapack::MatrixType::Upper, L-1, L-1, 0.0, 0.0, &B_ref[(n-L+1)*ldb], ldb );

        print_matrix( "Aref",  m, m,   &A_ref[0], lda, verbose );
        print_matrix( "Bref1", m, n-L, &B_ref[0], ldb, verbose );
        print_matrix( "Bref2", m, L,   &B_ref[(n-L)*ldb], ldb, verbose );

        // [ LA LB ] = [ LA LB ] * Q
        info_tst = lapack::tpmlqt(
            lapack::Side::Right, lapack::Op::NoTrans, m, n, m, L, mb,
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

        print_matrix( "A_QR0", m, m,   &A_ref[0],   lda, verbose );
        print_matrix( "A_QR1", m, n-L, &B_ref[0],   ldb, verbose );
        print_matrix( "A_QR2", m, L,   &B_ref[m-L], ldb, verbose );
    }

    if (params.ref() == 'y') {
        // ---------- run reference
        // Reset A, B in case check wiped them out.
        lapack::larnv( idist, iseed, A_ref.size(), &A_ref[0] );
        lapack::larnv( idist, iseed, B_ref.size(), &B_ref[0] );
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_tplqt(
            m, n, L, mb, &A_ref[0], lda, &B_ref[0], ldb, &T_ref[0], ldt );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_tplqt returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_tplqt( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_tplqt_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_tplqt_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_tplqt_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_tplqt_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}

#else

// -----------------------------------------------------------------------------
void test_tplqt( Params& params, bool run )
{
    fprintf( stderr, "tplqt requires LAPACK >= 3.7.0\n\n" );
    exit(0);
}

#endif  // LAPACK >= 3.7.0
