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

#if LAPACK_VERSION >= 30400  // >= 3.4.0

//------------------------------------------------------------------------------
// Simple overloaded wrappers around LAPACKE (assuming routines in LAPACKE).
// These should go in test/lapacke_wrappers.hh.
inline lapack_int LAPACKE_gemqrt(
    char side, char trans, lapack_int m, lapack_int n, lapack_int k, lapack_int nb,
    float* V, lapack_int ldv,
    float* T, lapack_int ldt,
    float* C, lapack_int ldc )
{
    return LAPACKE_sgemqrt(
        LAPACK_COL_MAJOR, side, trans, m, n, k, nb,
        V, ldv,
        T, ldt,
        C, ldc );
}

inline lapack_int LAPACKE_gemqrt(
    char side, char trans, lapack_int m, lapack_int n, lapack_int k, lapack_int nb,
    double* V, lapack_int ldv,
    double* T, lapack_int ldt,
    double* C, lapack_int ldc )
{
    return LAPACKE_dgemqrt(
        LAPACK_COL_MAJOR, side, trans, m, n, k, nb,
        V, ldv,
        T, ldt,
        C, ldc );
}

inline lapack_int LAPACKE_gemqrt(
    char side, char trans, lapack_int m, lapack_int n, lapack_int k, lapack_int nb,
    std::complex<float>* V, lapack_int ldv,
    std::complex<float>* T, lapack_int ldt,
    std::complex<float>* C, lapack_int ldc )
{
    return LAPACKE_cgemqrt(
        LAPACK_COL_MAJOR, side, trans, m, n, k, nb,
        (lapack_complex_float*) V, ldv,
        (lapack_complex_float*) T, ldt,
        (lapack_complex_float*) C, ldc );
}

inline lapack_int LAPACKE_gemqrt(
    char side, char trans, lapack_int m, lapack_int n, lapack_int k, lapack_int nb,
    std::complex<double>* V, lapack_int ldv,
    std::complex<double>* T, lapack_int ldt,
    std::complex<double>* C, lapack_int ldc )
{
    return LAPACKE_zgemqrt(
        LAPACK_COL_MAJOR, side, trans, m, n, k, nb,
        (lapack_complex_double*) V, ldv,
        (lapack_complex_double*) T, ldt,
        (lapack_complex_double*) C, ldc );
}

//------------------------------------------------------------------------------
template< typename scalar_t >
void test_gemqrt_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Side side = params.side();
    lapack::Op trans = params.trans();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    int64_t nb = params.nb();
    int64_t align = params.align();

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();
    params.gflops();

    if (! run)
        return;

    //---------- setup
    int64_t ldv;
    if (side == lapack::Side::Right) {
        ldv = roundup( blas::max( 1, m ), align );
    } else {
        ldv = roundup( blas::max( 1, n ), align );
    }
    int64_t ldt = roundup( nb, align );
    int64_t ldc = roundup( blas::max( 1, m ), align );
    size_t size_V = (size_t) ldv * k;
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
    lapack::larnv( idist, iseed, C_tst.size(), &C_tst[0] );
    C_ref = C_tst;

    //---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::gemqrt( side, trans, m, n, k, nb, &V[0], ldv, &T[0], ldt, &C_tst[0], ldc );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gemqrt returned error %lld\n", (lld) info_tst );
    }

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::gemqrt( side, trans, m, n, k, nb );
    //params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        //---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_gemqrt( side2char(side), op2char(trans), m, n, k, nb, &V[0], ldv, &T[0], ldt, &C_ref[0], ldc );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gemqrt returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;

        //---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( C_tst, C_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

#endif  // LAPACK >= 3.4.0

//------------------------------------------------------------------------------
void test_gemqrt( Params& params, bool run )
{
#if LAPACK_VERSION >= 30400  // >= 3.4.0
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gemqrt_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gemqrt_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gemqrt_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gemqrt_work< std::complex<double> >( params, run );
            break;
    }
#else
    fprintf( stderr, "gemqrt requires LAPACK >= 3.4.0\n\n" );
    exit(0);
#endif
}