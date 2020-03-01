// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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
template< typename scalar_t >
void test_tplqt_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t l = params.l();
    int64_t mb = params.nb();  // rename nb => mb
    int64_t align = params.align();

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();
    params.gflops();
    params.msg();

    if (! run)
        return;

    // skip invalid sizes
    if (blas::min(m, n) < l || m < mb || mb < 1) {
        params.msg() = "skipping: requires min(m, n) >= l and m >= mb >= 1";
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

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::tplqt( m, n, l, mb, &A_tst[0], lda, &B_tst[0], ldb, &T_tst[0], ldt );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::tplqt returned error %lld\n", (lld) info_tst );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::gelqf( m, n );  // estimate
    params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_tplqt( m, n, l, mb, &A_ref[0], lda, &B_ref[0], ldb, &T_ref[0], ldt );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_tplqt returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( B_tst, B_ref );
        error += abs_error( T_tst, T_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_tplqt( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

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
