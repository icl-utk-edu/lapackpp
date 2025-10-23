// Copyright (c) 2017-2025, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "lapack.hh"
#include "lapack/device.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_larfg_device_work( Params& params, bool run )
{
    using blas::real;
    using blas::imag;
    using lapack::device_info_int;
    using real_t = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t incx = params.incx();
    scalar_t alpha_tst = params.alpha.get<scalar_t>();
    scalar_t alpha_ref = alpha_tst;
    int64_t device = params.device();
    int64_t verbose = params.verbose();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol() * eps;

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();
    params.gflops();

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // ---------- setup
    scalar_t tau_tst;
    scalar_t tau_ref;
    size_t size_X = (size_t) (1+(n-2)*std::abs(incx));

    std::vector< scalar_t > X_tst( size_X );
    std::vector< scalar_t > X_ref( size_X );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, X_tst.size(), &X_tst[0] );
    X_ref = X_tst;

    // Allocate and copy to GPU
    lapack::Queue queue( device );
    scalar_t* dX_tst = blas::device_malloc< scalar_t >( size_X, queue );
    blas::device_copy_vector( size_X, X_tst.data(), 1, dX_tst, 1, queue );

    if (verbose >= 1) {
        printf( "x incx %lld, size %lld\n", llong( incx ), llong( size_X ) );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei\n", real(alpha_tst), imag(alpha_tst) );
        printf( "x = " ); print_vector( n-1, &X_tst[0], incx );
        printf( "xref = " ); print_vector( n-1, &X_ref[0], incx );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    queue.sync();
    double time = testsweeper::get_wtime();
    lapack::larfg( n, &alpha_tst, dX_tst, incx, &tau_tst, queue );
    queue.sync();
    time = testsweeper::get_wtime() - time;

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::larfg( n );
    params.gflops() = gflop / time;

    // Copy result back to CPU
    blas::device_copy_vector( size_X, dX_tst, 1, X_tst.data(), 1, queue );
    queue.sync();
    blas::device_free( dX_tst, queue );

    if (verbose >= 2) {
        printf( "alpha2 = %.4e\n", real(alpha_tst) );
        printf( "x2 = " ); print_vector( n-1, &X_tst[0], incx );
        printf( "tau = %.4e\n", real(tau_tst) );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_larfg( n, &alpha_ref, &X_ref[0], incx,
                                          &tau_ref );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_larfg returned error %lld\n",
                     llong( info_ref ) );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "alpha2ref = %.4e\n", real(alpha_ref) );
            printf( "x2ref = " ); print_vector( n-1, &X_ref[0], incx );
            printf( "tau_ref = %.4e\n", real(tau_ref) );
        }

        // ---------- check error compared to reference
        real_t error = 0;
        error += std::abs( alpha_tst - alpha_ref ) / std::abs(alpha_ref);
        error += rel_error( X_tst, X_ref );
        error += std::abs( tau_tst - tau_ref ) / std::abs(tau_ref);
        params.error() = error;
        params.okay() = (error < tol);
    }
}

// -----------------------------------------------------------------------------
void test_larfg_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_larfg_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_larfg_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_larfg_device_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_larfg_device_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
