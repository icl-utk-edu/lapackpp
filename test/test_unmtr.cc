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
void test_unmtr_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // Constants
    real_t eps = std::numeric_limits<real_t>::epsilon();

    // get & mark input values
    lapack::Side side = params.side();
    lapack::Uplo uplo = params.uplo();
    lapack::Op trans = params.trans();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align();
    real_t tol = params.tol() * eps;
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t r = (side == lapack::Side::Left) ? m : n;
    int64_t lda = roundup( blas::max( 1, r ), align );
    int64_t ldc = roundup( blas::max( 1, m ), align );
    size_t size_A = (size_t) ( blas::max( 1, lda*r ) );
    size_t size_tau = (size_t) ( blas::max( 1, r-1 ) );
    size_t size_C = (size_t) blas::max( 1, ldc * n );
    size_t size_D = (size_t) (r);
    size_t size_E = (size_t) (r-1);

    std::vector< scalar_t > A( size_A );
    std::vector< scalar_t > tau( size_tau );
    std::vector< scalar_t > C_tst( size_C );
    std::vector< scalar_t > C_ref( size_C );
    std::vector< real_t > D( size_D );
    std::vector< real_t > E( size_E );

    lapack::generate_matrix( params.matrix, r, r, &A[0], lda );
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, tau.size(), &tau[0] );
    lapack::larnv( idist, iseed, C_tst.size(), &C_tst[0] );
    C_ref = C_tst;

    int64_t info = lapack::hetrd( uplo, r, &A[0], lda, &D[0], &E[0], &tau[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::hetrd returned error %lld\n", llong( info ) );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::unmtr( side, uplo, trans, m, n, &A[0], lda, &tau[0], &C_tst[0], ldc );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::unmtr returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::unmtr( side, trans, m, n );
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_unmtr( side2char(side), uplo2char(uplo), op2char(trans), m, n, &A[0], lda, &tau[0], &C_ref[0], ldc );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_unmtr returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error = blas::max( error, rel_error( C_tst, C_ref ) );
        params.error() = error;
        params.okay() = (error < tol);
    }
}

// -----------------------------------------------------------------------------
void test_unmtr( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_unmtr_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_unmtr_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_unmtr_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_unmtr_work< std::complex<double> >( params, run );
            break;
    }
}
