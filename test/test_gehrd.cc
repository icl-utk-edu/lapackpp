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

#include "check_gehrd.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gehrd_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t ilo = 1;
    int64_t ihi = n;
    int64_t align = params.align();
    int64_t verbose = params.verbose();
    params.matrix.mark();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol() * eps;

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();
    params.gflops();
    params.ortho();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, n ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_tau = (size_t) (n-1);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > tau_tst( size_tau );
    std::vector< scalar_t > tau_ref( size_tau );

    lapack::generate_matrix( params.matrix, n, n, &A_tst[0], lda );
    A_ref = A_tst;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::gehrd( n, ilo, ihi, &A_tst[0], lda, &tau_tst[0] );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gehrd returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::gehrd( n );
    params.gflops() = gflop / time;

    if (params.check() == 'y') {
        // ---------- check numerical error
        real_t results[2];
        check_gehrd( n, &A_ref[0], lda, &A_tst[0], lda, &tau_tst[0],
                     verbose, results );
        params.error() = results[0];
        params.ortho() = results[1];
        params.okay() = (results[0] < tol && results[1] < tol);
    }

    if (params.ref() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_gehrd( n, ilo, ihi, &A_ref[0], lda, &tau_ref[0] );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gehrd returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_gehrd( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gehrd_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gehrd_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gehrd_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gehrd_work< std::complex<double> >( params, run );
            break;
    }
}
