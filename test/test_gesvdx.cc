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

#if LAPACK_VERSION >= 30600  // >= 3.6.0

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gesvdx_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    lapack::Job jobu = params.jobz();
    lapack::Job jobvt = params.jobvr();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align();

    real_t  vl;  // = params.vl();
    real_t  vu;  // = params.vu();
    int64_t il;  // = params.il();
    int64_t iu;  // = params.iu();
    lapack::Range range;  // derived from vl,vu,il,iu
    params.get_range( n, &range, &vl, &vu, &il, &iu );
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();
    params.msg();

    if (! run)
        return;

    // skip invalid sizes
    if (range == lapack::Range::Index
        && ! (1 <= il && il < iu && iu < blas::min( m, n ))) {
        params.msg() = "skipping: requires 1 <= il <= iu <= min(m,n)";
        return;
    }

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, m ), align );
    int64_t ns_tst;
    lapack_int ns_ref;
    int64_t ldu = ( jobu == lapack::Job::Vec ? roundup( m, align ) : 1 );
    int64_t ldvt = ( jobvt == lapack::Job::Vec ? roundup( blas::min( m, n ), align ) : 1 );
    size_t size_A = (size_t) ( lda * n );
    size_t size_S = (size_t) ( blas::min( m, n) );
    size_t size_U = (size_t) ( ldu * blas::min( m, n ) );
    size_t size_VT = (size_t) ( ldvt * n );

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< real_t > S_tst( size_S );
    std::vector< real_t > S_ref( size_S );
    std::vector< scalar_t > U_tst( size_U );
    std::vector< scalar_t > U_ref( size_U );
    std::vector< scalar_t > VT_tst( size_VT );
    std::vector< scalar_t > VT_ref( size_VT );

    lapack::generate_matrix( params.matrix, m, n, &A_tst[0], lda );
    A_ref = A_tst;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::gesvdx( jobu, jobvt, range, m, n, &A_tst[0], lda, vl, vu, il, iu, &ns_tst, &S_tst[0], &U_tst[0], ldu, &VT_tst[0], ldvt );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gesvdx returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::gesvdx( jobu, jobvt, range, m, n );
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_gesvdx( job2char(jobu), job2char(jobvt), range2char(range), m, n, &A_ref[0], lda, vl, vu, il, iu, &ns_ref, &S_ref[0], &U_ref[0], ldu, &VT_ref[0], ldvt );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gesvdx returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += std::abs( ns_tst - ns_ref );
        error += abs_error( S_tst, S_ref );
        error += abs_error( U_tst, U_ref );
        error += abs_error( VT_tst, VT_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gesvdx( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gesvdx_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gesvdx_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gesvdx_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gesvdx_work< std::complex<double> >( params, run );
            break;
    }
}

#else

// -----------------------------------------------------------------------------
void test_gesvdx( Params& params, bool run )
{
    fprintf( stderr, "gesvdx requires LAPACK >= 3.6.0\n\n" );
    exit(0);
}

#endif  // LAPACK >= 3.6.0
