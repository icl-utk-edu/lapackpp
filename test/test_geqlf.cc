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
void test_geqlf_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align();
    params.matrix.mark();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol() * eps;

    // mark non-standard output values
    params.ortho();
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, m ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_tau = (size_t) (blas::min(m,n));

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > tau_tst( size_tau );
    std::vector< scalar_t > tau_ref( size_tau );

    lapack::generate_matrix( params.matrix, m, n, &A_tst[0], lda );
    A_ref = A_tst;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::geqlf( m, n, &A_tst[0], lda, &tau_tst[0] );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::geqlf returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::geqlf( m, n );
    params.gflops() = gflop / time;

    if (params.check() == 'y') {
        // ---------- check error
        // comparing to ref. solution doesn't work
        // Following magma/testing/testing_zgeqlf.cpp
        int64_t minmn = blas::min( m, n );

        int64_t ldq = m;
        int64_t ldl = minmn;
        std::vector< scalar_t > Q( ldq * minmn ); // m by k
        std::vector< scalar_t > L( ldl * n ); // k by n

        // copy M by K matrix V to Q (copying diagonal, which isn't needed)
        // copy K by N matrix L
        lapack::laset( lapack::MatrixType::General, minmn, n, 0, 0, &L[0], ldl );
        if (m >= n) {
            int64_t m_n = m - n;
            lapack::lacpy( lapack::MatrixType::General, m_n, minmn, &A_tst[0], lda, &Q[0], ldq );
            lapack::lacpy( lapack::MatrixType::Upper, n, minmn, &A_tst[m_n], lda, &Q[m_n], ldq );
            lapack::lacpy( lapack::MatrixType::Lower, minmn, n, &A_tst[m_n], lda, &L[0], ldl );
        }
        else {
            int64_t n_m = n - m;
            lapack::lacpy( lapack::MatrixType::Upper, m, minmn, &A_tst[n_m*lda], lda, &Q[0], ldq );
            lapack::lacpy( lapack::MatrixType::General, minmn, n_m, &A_tst[0], lda, &L[0], ldl );
            lapack::lacpy( lapack::MatrixType::Lower, minmn, m, &A_tst[n_m*lda], lda, &L[n_m*ldl], ldl );
        }

        // generate M by K matrix Q, where K = min(M,N)
        int64_t info_ungql = lapack::ungql( m, minmn, minmn, &Q[0], ldq, &tau_tst[0] );
        if (info_ungql != 0) {
            fprintf( stderr, "lapack::ungqr returned error %lld\n", llong( info_ungql ) );
        }

        // compute L - Q'*A
        blas::gemm( blas::Layout::ColMajor,
                    blas::Op::ConjTrans, blas::Op::NoTrans, minmn, n, m,
                    -1.0, &Q[0], ldq, &A_ref[0], lda, 1.0, &L[0], ldl );

        // error = || L - Q^H*A || / (N * ||A||)
        real_t Anorm = lapack::lange( lapack::Norm::One, m, n, &A_ref[0], lda );
        real_t resid1 = lapack::lange( lapack::Norm::One, minmn, n, &L[0], ldl );
        real_t error1 = 0;
        if (Anorm > 0)
            error1 = resid1 / ( n * Anorm );

        // set L = I (K by K identity), then L = I - Q^H*Q
        lapack::laset( lapack::MatrixType::Upper, minmn, minmn, 0.0, 1.0, &L[0], ldl );
        blas::herk( blas::Layout::ColMajor,
                    blas::Uplo::Upper, blas::Op::ConjTrans, minmn, m,
                    -1.0, &Q[0], ldq, 1.0, &L[0], ldl );

        // error = || I - Q^H*Q || / N
        real_t resid2 = lapack::lanhe( lapack::Norm::One, lapack::Uplo::Upper, minmn, &L[0], ldl );
        real_t error2 = ( resid2 / n );

        params.error() = error1;
        params.ortho() = error2;
        params.okay() = (error1 < tol) && (error2 < tol);
    }

    if (params.ref() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_geqlf( m, n, &A_ref[0], lda, &tau_ref[0] );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_geqlf returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_geqlf( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_geqlf_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_geqlf_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_geqlf_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_geqlf_work< std::complex<double> >( params, run );
            break;
    }
}
