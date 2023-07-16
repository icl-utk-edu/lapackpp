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
#include "check_gels.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gelss_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t align = params.align();
    params.matrix.mark();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol() * eps;

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();
    params.error2();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, m ), align );
    int64_t ldb = roundup( blas::max( 1, m, n ), align );
    real_t rcond = -1;  // use machine epsilon
    int64_t rank_tst;
    lapack_int rank_ref;
    size_t size_A = (size_t) lda * n;
    size_t size_B = (size_t) ldb * nrhs;
    size_t size_S = (size_t) (blas::min(m,n));

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );
    std::vector< real_t > S_tst( size_S );
    std::vector< real_t > S_ref( size_S );

    lapack::generate_matrix( params.matrix, m, n, &A_tst[0], lda );
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );

    A_ref = A_tst;
    B_ref = B_tst;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::gelss( m, n, nrhs, &A_tst[0], lda, &B_tst[0], ldb, &S_tst[0], rcond, &rank_tst );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gelss returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::gelss( m, n, nrhs );
    // params.gflops() = gflop / time;

    if (params.check() == 'y') {
        // ---------- check error
        real_t error[2];
        check_gels( false, blas::Op::NoTrans, m, n, nrhs,
                    &A_ref[0], lda, // original A
                    &B_tst[0], ldb, // X
                    &B_ref[0], ldb, // original B
                    error );
        params.error()  = error[0];
        params.error2() = error[1];
        params.okay() = (error[0] < tol) && (error[1] < tol);
    }

    if (params.ref() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_gelss( m, n, nrhs, &A_ref[0], lda, &B_ref[0], ldb, &S_ref[0], rcond, &rank_ref );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gelss returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_gelss( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gelss_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gelss_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gelss_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gelss_work< std::complex<double> >( params, run );
            break;
    }
}
