// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "lapack.hh"
#include "lapack/device.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "check_heev.hh"
#include "lapacke_wrappers.hh"

#include <vector>

template< typename scalar_t >
void test_heevd_device_work( Params& params, bool run )
{
    using lapack::device_info_int;
    using real_t = blas::real_type< scalar_t >;
    using lapack::Job;

    // Constants
    const real_t eps = std::numeric_limits< real_t >::epsilon();

    // get & mark input values
    lapack::Job jobz = params.jobz();
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t device = params.device();
    int64_t align = params.align();
    int64_t verbose = params.verbose();
    real_t tol = params.tol() * eps;
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();
    params.ortho();
    params.error2();
    params.error2.name( "Lambda" );

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, n ), align );
    int64_t ldz = lda;  // vectors overwrite matrix A
    size_t size_A = (size_t) lda * n;
    size_t size_Z = size_A;
    size_t size_W = (size_t) n;

    std::vector< scalar_t > A( size_A );
    std::vector< scalar_t > Z( size_Z );  // eigenvectors
    std::vector< real_t > Lambda_tst( n );
    std::vector< real_t > Lambda_ref( n );

    lapack::generate_matrix( params.matrix, n, n, &A[0], lda );
    Z = A;

    // Allocate and copy to GPU
    lapack::Queue queue( device );
    scalar_t*        dA_tst = blas::device_malloc< scalar_t >( size_A, queue );
    real_t*          dW_tst = blas::device_malloc< real_t >  ( size_W, queue );
    device_info_int* d_info = blas::device_malloc< device_info_int >( 1, queue );
    blas::device_copy_matrix( n, n, A.data(), lda, dA_tst, lda, queue );


    // Allocate workspace
    size_t d_size, h_size;
    lapack::heevd_work_size_bytes( jobz, uplo, n, dA_tst, lda, dW_tst,
                                   &d_size, &h_size, queue );
    char* d_work = blas::device_malloc< char >( d_size, queue );
    std::vector<char> h_work_vector( h_size );
    char* h_work = h_work_vector.data();


    if (verbose >= 1) {
        printf( "\n" );
        printf( "A n=%5lld, lda=%5lld\n", llong( n ), llong( lda ) );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( n, n, &A[0], lda );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    queue.sync();
    double time = testsweeper::get_wtime();

    lapack::heevd( jobz, uplo, n, dA_tst, lda, dW_tst, d_work, d_size,
                   h_work, h_size, d_info, queue );

    queue.sync();
    time = testsweeper::get_wtime() - time;

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::heev( jobz, n );
    // params.gflops() = gflop / time;

    // Copy result back to CPU.
    device_info_int info_tst;
    blas::device_copy_matrix( n, n, dA_tst, lda, Z.data(), ldz, queue );
    blas::device_copy_vector( n, dW_tst, 1, Lambda_tst.data(), 1, queue );
    blas::device_memcpy( &info_tst, d_info, 1, queue );
    queue.sync();


    if (info_tst != 0) {
        fprintf( stderr, "lapack::heev returned error %lld\n", llong( info_tst ) );
    }

    // Cleanup GPU memory
    blas::device_free( dA_tst, queue );
    blas::device_free( dW_tst, queue );
    blas::device_free( d_work, queue );
    blas::device_free( d_info, queue );

    if (verbose >= 2) {
        printf( "Z = " ); print_matrix( n, n, &Z[0], ldz );
        printf( "Lambda = " ); print_vector( n, &Lambda_tst[0], 1 );
    }

    if (params.check() == 'y') {
        // ---------- check numerical error
        // result[ 0 ] = || A - Z Lambda Z^H || / (n ||A||), if jobz != NoVec.
        // result[ 1 ] = || I - Z^H Z || / n, if jobz != NoVec.
        // result[ 2 ] = 0 if Lambda is in non-decreasing order, else > 0.
        real_t result[ 3 ] = { (real_t) testsweeper::no_data_flag,
                               (real_t) testsweeper::no_data_flag,
                               (real_t) testsweeper::no_data_flag };

        check_heev( jobz, uplo, n, &A[0], lda,
                    n, &Lambda_tst[0], &Z[0], ldz, result );

        params.error()  = result[ 0 ];
        params.ortho()  = result[ 1 ];
        params.error2() = result[ 2 ];
        params.okay()   = (jobz == Job::NoVec || result[ 0 ] < tol)
                       && (jobz == Job::NoVec || result[ 1 ] < tol)
                       && result[ 2 ] < tol;
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_heev(
            to_char( jobz ), to_char( uplo ), n,
            &A[0], lda, &Lambda_ref[0] );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_heev returned error %lld\n", llong( info_ref ) );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = rel_error( Lambda_tst, Lambda_ref );
        if (info_tst != info_ref) {
            error = 1;
        }
        params.error2() = error;
        params.okay() = params.okay() && (error < tol);
    }
}

// -----------------------------------------------------------------------------
void test_heevd_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_heevd_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_heevd_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_heevd_device_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_heevd_device_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
