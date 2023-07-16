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
void test_larfb_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Side side = params.side();
    lapack::Op trans = params.trans();
    lapack::Direction direction = params.direction();
    lapack::StoreV storev = params.storev();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    int64_t align = params.align();

    // mark non-standard output values
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();
    params.msg();

    if (! run)
        return;

    // skip invalid sizes
    if ((side == lapack::Side::Left  && m < k) ||
        (side == lapack::Side::Right && n < k))
    {
        params.msg() = "skipping: requires m >= k >= 0 (left) or n >= k >= 0 (right)";
        return;
    }

    // skip invalid configuration
    if ((blas::is_complex<scalar_t>::value) &&
        (trans == lapack::Op::Trans))
    {
        params.msg() = "skipping: requires Op::NoTrans or Op::ConjTrans if complex";
        return;
    }

    // ---------- setup
    int64_t ldv;
    if (storev == lapack::StoreV::Columnwise) {
        if (side == lapack::Side::Left)
            ldv = roundup( blas::max( 1, m ), align );
        else
            ldv = roundup( blas::max( 1, n ), align );
    }
    else {
        // rowwise
        ldv = roundup( k, align );
    }

    int64_t ldt = roundup( k, align );
    int64_t ldc = roundup( blas::max( 1, m ), align );

    size_t size_V;
    if (storev == lapack::StoreV::Columnwise) {
        size_V = (size_t) ldv * k;
    }
    else {
        // rowwise
        if (side == lapack::Side::Left)
            size_V = (size_t) ldv * m;
        else
            size_V = (size_t) ldv * n;
    }

    size_t size_T = (size_t) ldt * k;
    size_t size_C = (size_t) ldc * n;

    std::vector< scalar_t > V( size_V );
    std::vector< scalar_t > T( size_T );
    std::vector< scalar_t > C_tst( size_C );
    std::vector< scalar_t > C_ref( size_C );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, V.size(), &V[0] );
    lapack::larnv( idist, iseed, T.size(), &T[0] );
    lapack::generate_matrix( params.matrix, m, n, &C_tst[0], ldc );
    C_ref = C_tst;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    lapack::larfb( side, trans, direction, storev, m, n, k, &V[0], ldv, &T[0], ldt, &C_tst[0], ldc );
    time = testsweeper::get_wtime() - time;

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::larfb( side, trans, direction, storev, m, n, k );
    //params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_larfb( side2char(side), op2char(trans), direction2char(direction), storev2char(storev), m, n, k, &V[0], ldv, &T[0], ldt, &C_ref[0], ldc );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_larfb returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;
        real_t tol = std::numeric_limits< real_t >::epsilon();

        // ---------- check error compared to reference
        real_t error = 0;
        error += rel_error( C_tst, C_ref );
        params.error() = error;
        params.okay() = (error <= tol);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_larfb( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_larfb_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_larfb_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_larfb_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_larfb_work< std::complex<double> >( params, run );
            break;
    }
}
