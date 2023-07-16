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
#include "scale.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_heevx_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;

    // Constants
    const scalar_t one  = 1.0;
    const real_t   eps  = std::numeric_limits< real_t >::epsilon();

    // get & mark input values
    lapack::Job jobz = params.jobz();
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t align = params.align();
    int64_t verbose = params.verbose();
    real_t tol = params.tol() * eps;
    params.matrix.mark();

    real_t  vl;  // = params.vl();
    real_t  vu;  // = params.vu();
    int64_t il;  // = params.il();
    int64_t iu;  // = params.iu();
    lapack::Range range;  // derived from vl,vu,il,iu
    params.get_range( n, &range, &vl, &vu, &il, &iu );

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();
    params.error2();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, n ), align );
    real_t abstol = 0;  // default value
    int64_t nfound;
    lapack_int nfound_ref;
    int64_t ldz = (jobz == lapack::Job::Vec
                   ? roundup( blas::max( 1, n ), align )
                   : 1 );
    size_t size_A = (size_t) lda * n;
    size_t size_Z = (size_t) ldz * n;

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > Z( size_Z );  // eigenvectors
    std::vector< real_t > Lambda_tst( n );
    std::vector< real_t > Lambda_ref( n );
    std::vector< int64_t > ifail_tst( n );
    std::vector< lapack_int > ifail_ref( n );

    lapack::generate_matrix( params.matrix, n, n, &A_tst[0], lda );
    A_ref = A_tst;

    if (verbose >= 1) {
        printf( "\n" );
        printf( "A n=%5lld, lda=%5lld\n", llong( n ), llong( lda ) );
    }
    if (verbose >= 2) {
        printf( "A = " );
        print_matrix( n, n, &A_tst[0], lda );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::heevx(
                           jobz, range, uplo, n,
                           &A_tst[0], lda,
                           vl, vu, il, iu, abstol, &nfound,
                           &Lambda_tst[0], &Z[0], ldz, &ifail_tst[0] );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::heevx returned error %lld\n", llong( info_tst ) );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::heevx( jobz, range, n );
    // params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "nfound = %lld\n", llong( nfound ) );
        printf( "Lambda = " );
        print_vector( n, &Lambda_tst[0], 1 );
        if (jobz == lapack::Job::Vec) {
            printf( "Z = " );
            print_matrix( n, nfound, &Z[0], ldz );
        }
    }

    if (params.check() == 'y' && jobz == lapack::Job::Vec) {
        // ---------- check error
        // Relative backwards error =
        //     ||A Z - Z Lambda|| / (n * ||A|| * ||Z||)
        real_t Anorm = lapack::lanhe( lapack::Norm::One, uplo, n, &A_ref[0], lda );
        real_t Znorm = lapack::lange( lapack::Norm::One, n, nfound, &Z[0], ldz );

        real_t error = 0;
        std::vector< scalar_t > W( size_Z );  // workspace
        int64_t ldw = ldz;
        // W = Z
        lapack::lacpy( lapack::MatrixType::General, n, nfound,
                       &Z[0], ldz,
                       &W[0], ldw );
        // W = Z Lambda
        col_scale( n, nfound, &W[0], ldw, &Lambda_tst[0] );
        // W = A Z - (Z Lambda)
        blas::hemm( blas::Layout::ColMajor, blas::Side::Left, uplo, n, nfound,
                    one,  &A_ref[0], lda,
                    &Z[0], ldz,
                    -one, &W[0], ldw );
        error = lapack::lange( lapack::Norm::One, n, nfound, &W[0], ldw );

        if (verbose >= 2) {
            printf( "W = " );
            print_matrix( n, nfound, &W[0], ldw );
        }

        error /= (n * Anorm * Znorm);
        params.error() = error;
        params.okay() = (error < tol);
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_heevx(
                               job2char(jobz), range2char(range), uplo2char(uplo), n,
                               &A_ref[0], lda,
                               vl, vu, il, iu, abstol, &nfound_ref,
                               &Lambda_ref[0], &Z[0], ldz, &ifail_ref[0] );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_heevx returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "Lambda_ref" );
            print_vector( n, &Lambda_ref[0], 1 );
            if (jobz == lapack::Job::Vec) {
                printf( "Zref" );
                print_matrix( n, nfound, &Z[0], ldz );
            }
        }

        // ---------- check error compared to reference
        real_t error = rel_error( Lambda_tst, Lambda_ref );
        if (info_tst != info_ref) {
            error = 1;
        }
        error += std::abs( nfound - nfound_ref );
        if (jobz == lapack::Job::Vec) {
            // Check first nfound elements of ifail
            for (int64_t i = 0; i < nfound; ++i)
                error += std::abs( ifail_tst[i] - ifail_ref[i] );
        }
        params.error2() = error;
        params.okay() = params.okay() && (error < tol);
    }
}

// -----------------------------------------------------------------------------
void test_heevx( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_heevx_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_heevx_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_heevx_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_heevx_work< std::complex<double> >( params, run );
            break;
    }
}
