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
void test_gglse_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    // TODO int64_t p = params.p();
    int64_t p = params.dim.k();
    int64_t align = params.align();
    params.matrix.mark();
    params.matrixB.mark();
    int64_t verbose = params.verbose();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol() * eps;

    // mark non-standard output values
    params.error2();
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();
    params.msg();

    if (! run) {
        // Use well-conditioned matrices per LAWN 41.
        params.matrix .kind.set_default( "svd" );
        params.matrixB.kind.set_default( "svd" );
        params.matrix .cond() = 100;
        params.matrixB.cond() = 10;
        return;
    }

    // skip invalid sizes
    if (! ((0 <= p) && (p <= n) && (n <= m+p))) {
        params.msg() = "skipping: requires 0 <= p <= n <= m+p";
        return;
    }

    // ---------- setup
    bool consistent = true;
    int64_t lda = roundup( blas::max( 1, m ), align );
    int64_t ldb = roundup( blas::max( 1, p ), align );
    size_t size_A = (size_t) ( lda * n );
    size_t size_B = (size_t) ( ldb * n );
    size_t size_C = (size_t) (m);
    size_t size_D = (size_t) (p);
    size_t size_X = (size_t) (n);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );
    std::vector< scalar_t > C_tst( size_C );
    std::vector< scalar_t > C_ref( size_C );
    std::vector< scalar_t > D_tst( size_D );
    std::vector< scalar_t > D_ref( size_D );
    std::vector< scalar_t > X_tst( size_X );
    std::vector< scalar_t > X_ref( size_X );

    lapack::generate_matrix( params.matrix,  m, n, &A_tst[0], lda );
    lapack::generate_matrix( params.matrixB, p, n, &B_tst[0], ldb );
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    if (consistent) {
        // Generate random X, then set C = A*X and D = B*X.
        lapack::larnv( idist, iseed, X_ref.size(), &X_ref[0] );
        blas::gemv( blas::Layout::ColMajor, blas::Op::NoTrans, m, n,
                    1.0, &A_tst[0], lda, &X_ref[0], 1,
                    0.0, &C_tst[0], 1 );
        blas::gemv( blas::Layout::ColMajor, blas::Op::NoTrans, p, n,
                    1.0, &B_tst[0], ldb, &X_ref[0], 1,
                    0.0, &D_tst[0], 1 );
    }
    else {
        // Generate random C, D.
        lapack::larnv( idist, iseed, C_tst.size(), &C_tst[0] );
        lapack::larnv( idist, iseed, D_tst.size(), &D_tst[0] );
    }

    A_ref = A_tst;
    B_ref = B_tst;
    C_ref = C_tst;
    D_ref = D_tst;

    if (verbose >= 2) {
        printf( "A = " ); print_matrix( m, n, &A_tst[0], lda );
        printf( "B = " ); print_matrix( p, n, &B_tst[0], ldb );
        printf( "c = " ); print_vector( m, &C_tst[0], 1 );
        printf( "d = " ); print_vector( p, &D_tst[0], 1 );
    }

    // ---------- run test
    // minimize || c - A*x ||_2   subject to   B*x = d
    // A is M-by-N matrix, B is P-by-N matrix, c is M-vector, and d is P-vector
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::gglse( m, n, p, &A_tst[0], lda, &B_tst[0], ldb, &C_tst[0], &D_tst[0], &X_tst[0] );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gglse returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::gglse( m, n );
    // params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "x_tst = " ); print_vector( n, &X_tst[0], 1 );
    }

    if (params.check() == 'y') {
        // ---------- check error
        real_t x_norm = blas::asum( n, &X_tst[0], 1 );

        // r1 = Ax - c is small only for a consistent system.
        if (consistent) {
            C_tst = C_ref;
            blas::gemv( blas::Layout::ColMajor, blas::Op::NoTrans, m, n,
                         1.0, &A_ref[0], lda, &X_tst[0], 1,
                        -1.0, &C_tst[0], 1 );
            real_t error1 = blas::asum( m, &C_tst[0], 1 );
            real_t A_norm = lapack::lange( lapack::Norm::One, m, n, &A_ref[0], lda );
            if (A_norm != 0)
                error1 /= A_norm;
            if (x_norm != 0)
                error1 /= x_norm;

            params.error() = error1;
            params.okay() = (error1 < tol);
        }

        // r2 = Bx - d should always be small.
        D_tst = D_ref;
        blas::gemv( blas::Layout::ColMajor, blas::Op::NoTrans, p, n,
                     1.0, &B_ref[0], ldb, &X_tst[0], 1,
                    -1.0, &D_tst[0], 1 );
        real_t error2 = blas::asum( p, &D_tst[0], 1 );
        real_t B_norm = lapack::lange( lapack::Norm::One, p, n, &B_ref[0], ldb );
        error2 /= n;
        if (B_norm != 0)
            error2 /= B_norm;
        if (x_norm != 0)
            error2 /= x_norm;

        params.error2() = error2;
        params.okay() = params.okay() && (error2 < tol);
    }

    if (params.ref() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_gglse( m, n, p, &A_ref[0], lda, &B_ref[0], ldb, &C_ref[0], &D_ref[0], &X_ref[0] );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gglse returned error %lld\n", llong( info_ref ) );
        }

        if (verbose >= 2) {
            printf( "x_ref = " ); print_vector( n, &X_ref[0], 1 );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_gglse( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gglse_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gglse_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gglse_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gglse_work< std::complex<double> >( params, run );
            break;
    }
}
