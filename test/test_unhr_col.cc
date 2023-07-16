// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "lapack.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

#if LAPACK_VERSION >= 30900  // >= 3.9.0

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_unhr_col_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nb = params.nb();
    int64_t align = params.align();

    // mark non-standard output values
    params.ref_time();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, m ), align );
    int64_t ldt = roundup( blas::max( 1, blas::min( nb, n ) ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_T = (size_t) ldt * n;
    size_t size_D = (size_t) blas::min( m, n );

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > T_tst( size_T );
    std::vector< scalar_t > T_ref( size_T );
    std::vector< scalar_t > D_tst( size_D );
    std::vector< scalar_t > D_ref( size_D );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    A_ref = A_tst;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::unhr_col( m, n, nb, &A_tst[0], lda, &T_tst[0], ldt, &D_tst[0] );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::unhr_col returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;

    #ifdef LAPACK_HAVE_MKL
    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_unhr_col( m, n, nb, &A_ref[0], lda, &T_ref[0], ldt, &D_ref[0] );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_unhr_col returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( T_tst, T_ref );
        error += abs_error( D_tst, D_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
    #else
        // LAPACKE_unhr_col not yet in LAPACK
        params.msg() = "check requires Intel MKL, as of LAPACK 3.11";
    #endif  // LAPACK_HAVE_MKL
}

#endif  // LAPACK >= 3.9.0

// -----------------------------------------------------------------------------
void test_unhr_col( Params& params, bool run )
{
#if LAPACK_VERSION >= 30900  // >= 3.9.0
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_unhr_col_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_unhr_col_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_unhr_col_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_unhr_col_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
#else
    fprintf( stderr, "unhr_col requires LAPACK >= 3.9.0\n\n" );
    exit(0);
#endif
}
