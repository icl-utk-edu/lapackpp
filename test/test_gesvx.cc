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
void test_gesvx_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // Constants
    const scalar_t one = 1.0;
    const real_t   eps = std::numeric_limits< real_t >::epsilon();

    // get & mark input values
    lapack::Factored fact = params.factored();
    lapack::Op trans = params.trans();
    //lapack::Equed equed = params.equed();  // todo: pre or post multiply A
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t align = params.align();
    int64_t verbose = params.verbose();
    real_t tol = params.tol() * eps;
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();
    params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t lda  = roundup( blas::max( 1, n ), align );
    int64_t ldaf = roundup( blas::max( 1, n ), align );
    int64_t ldb  = roundup( blas::max( 1, n ), align );
    int64_t ldx  = roundup( blas::max( 1, n ), align );
    real_t rcond_tst = 0;
    real_t rcond_ref = 0;
    real_t rpivot_tst = 0;
    real_t rpivot_ref = 0;
    // equed is input if fact = 'f'; otherwise it is output.
    lapack::Equed equed_tst = lapack::Equed::None;
    lapack::Equed equed_ref = lapack::Equed::None;
    size_t size_A = (size_t) lda * n;
    size_t size_AF = (size_t) ldaf * n;
    size_t size_ipiv = (size_t) (n);
    size_t size_R = (size_t) (n);
    size_t size_C = (size_t) (n);
    size_t size_B = (size_t) ldb * nrhs;
    size_t size_X = (size_t) ldx * nrhs;
    size_t size_ferr = (size_t) (nrhs);
    size_t size_berr = (size_t) (nrhs);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > AF_tst( size_AF );
    std::vector< scalar_t > AF_ref( size_AF );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );
    std::vector< real_t > R_tst( size_R );
    std::vector< real_t > R_ref( size_R );
    std::vector< real_t > C_tst( size_C );
    std::vector< real_t > C_ref( size_C );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );
    std::vector< scalar_t > X_tst( size_X );
    std::vector< scalar_t > X_ref( size_X );
    std::vector< real_t > ferr_tst( size_ferr );
    std::vector< real_t > ferr_ref( size_ferr );
    std::vector< real_t > berr_tst( size_berr );
    std::vector< real_t > berr_ref( size_berr );

    lapack::generate_matrix( params.matrix, n, n, &A_tst[0], lda );
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, R_tst.size(), &R_tst[0] );
    lapack::larnv( idist, iseed, C_tst.size(), &C_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );

    // Factor A using copy AF to initialize ipiv_tst and ipiv_ref and AF
    AF_tst = A_tst;
    int64_t info_trf = lapack::getrf( n, n, &AF_tst[0], lda, &ipiv_tst[0] );
    if (info_trf != 0) {
        fprintf( stderr, "lapack::getrf returned error %lld\n", llong( info_trf ) );
    }
    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    A_ref = A_tst;
    AF_ref = AF_tst;
    R_ref = R_tst;
    C_ref = C_tst;
    B_ref = B_tst;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::gesvx(
                           fact, trans, n, nrhs,
                           &A_tst[0], lda,
                           &AF_tst[0], ldaf, &ipiv_tst[0],
                           &equed_tst, &R_tst[0], &C_tst[0],
                           &B_tst[0], ldb, &X_tst[0], ldx,
                           &rcond_tst, &ferr_tst[0], &berr_tst[0], &rpivot_tst );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gesvx returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    // todo: gflop isn't right if already factored (fact = 'f').
    double gflop = lapack::Gflop< scalar_t >::gesv( n, nrhs );
    params.gflops() = gflop / time;

    if (params.check() == 'y') {
        // ---------- check error
        // Relative backwards error = ||b - Ax|| / (n * ||A|| * ||x||).
        B_tst = B_ref;  // equed may have modified B.
        blas::gemm( blas::Layout::ColMajor, trans, blas::Op::NoTrans,
                    n, nrhs, n,
                    -one, &A_ref[0], lda,
                          &X_tst[0], ldb,
                    one,  &B_tst[0], ldb );
        if (verbose >= 2) {
            printf( "R = " );
            print_matrix( n, nrhs, &B_tst[0], ldb );
        }

        real_t error = lapack::lange( lapack::Norm::One, n, nrhs, &B_tst[0], ldb );
        real_t Xnorm = lapack::lange( lapack::Norm::One, n, nrhs, &X_tst[0], ldb );
        real_t Anorm = lapack::lange( lapack::Norm::One, n, n,    &A_ref[0], lda );
        error /= (n * Anorm * Xnorm);
        params.error() = error;
        params.okay() = (error < tol);
    }

    if (params.ref() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        char equed_ref_char = lapack::equed2char( equed_ref );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_gesvx(
                               factored2char(fact), op2char(trans), n, nrhs,
                               &A_ref[0], lda,
                               &AF_ref[0], ldaf, &ipiv_ref[0],
                               &equed_ref_char, &R_ref[0], &C_ref[0],
                               &B_ref[0], ldb, &X_ref[0], ldx,
                               &rcond_ref, &ferr_ref[0], &berr_ref[0], &rpivot_ref );
        time = testsweeper::get_wtime() - time;
        equed_ref = lapack::char2equed( equed_ref_char );
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gesvx returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_gesvx( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_gesvx_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gesvx_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gesvx_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gesvx_work< std::complex<double> >( params, run );
            break;
    }
}
