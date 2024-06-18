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

//------------------------------------------------------------------------------
template< typename scalar_t >
void test_lasr_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;
    using blas::max;
    using lapack::Side, lapack::Pivot, lapack::Direction;

    // Constants
    const real_t eps = std::numeric_limits< real_t >::epsilon();

    // get & mark input values
    lapack::Side side = params.side();
    lapack::Pivot pivot = params.pivot();
    lapack::Direction direction = params.direction();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align();
    int64_t verbose = params.verbose();
    real_t tol = params.tol() * eps;

    // mark non-standard output values
    params.ref_time();

    if (! run)
        return;

    //---------- setup
    int64_t lda = roundup( max( 1, m ), align );
    size_t size_C = (size_t) (side == Side::Left ? m-1 : n-1);
    size_t size_S = (size_t) (side == Side::Left ? m-1 : n-1);
    size_t size_A = (size_t) lda * n;

    std::vector< real_t > C( size_C );
    std::vector< real_t > S( size_S );
    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, C.size(), &C[0] );
    lapack::larnv( idist, iseed, S.size(), &S[0] );
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    A_ref = A_tst;

    if (verbose >= 2 ) {
        printf( "A = " ); print_matrix( m, n, &A_tst[0], lda );
        printf( "C = " ); print_vector( size_C, &C[0], 1 );
        printf( "S = " ); print_vector( size_C, &S[0], 1 );
    }

    //---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    lapack::lasr( side, pivot, direction, m, n, &C[0], &S[0], &A_tst[0], lda );
    time = testsweeper::get_wtime() - time;

    params.time() = time;

    if (verbose >= 2 ) {
        printf( "A_out = " );
        print_matrix( m, n, &A_tst[0], lda );
    }

    if (params.check() == 'y') {
        //---------- run reference, using rot
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        if (side == Side::Left) {
            // Left: update rows i0 and i1.
            int64_t begin, end, step;
            if (direction == Direction::Forward) {
                begin = 0;
                end   = m-1;
                step  = 1;
            }
            else {
                begin = m-2;
                end   = -1;
                step  = -1;
            }
            for (int64_t i = begin; step > 0 ? i < end : i > end; i += step) {
                int64_t i0=0, i1=0;
                switch (pivot) {
                    case Pivot::Top:      i0 = 0; i1 = i+1; break;
                    case Pivot::Bottom:   i0 = i; i1 = m-1; break;
                    case Pivot::Variable: i0 = i; i1 = i+1; break;
                }
                blas::rot( n, &A_ref[ i0 ], lda, &A_ref[ i1 ], lda,
                           C[ i ], S[ i ] );
            }
        }
        else {
            // Right: update cols j0 and j1.
            int64_t begin, end, step;
            if (direction == Direction::Forward) {
                begin = 0;
                end   = n-1;
                step  = 1;
            }
            else {
                begin = n-2;
                end   = -1;
                step  = -1;
            }
            for (int64_t j = begin; step > 0 ? j < end : j > end; j += step) {
                int64_t j0=0, j1=0;
                switch (pivot) {
                    case Pivot::Top:      j0 = 0; j1 = j+1; break;
                    case Pivot::Bottom:   j0 = j; j1 = n-1; break;
                    case Pivot::Variable: j0 = j; j1 = j+1; break;
                }
                blas::rot( m, &A_ref[ j0*lda ], 1, &A_ref[ j1*lda ], 1,
                           C[ j ], S[ j ] );
            }
        }
        time = testsweeper::get_wtime() - time;

        params.ref_time() = time;

        if (verbose >= 2 ) {
            printf( "A_ref = " );
            print_matrix( m, n, &A_ref[0], lda );
        }

        //---------- check error compared to reference
        real_t error = rel_error( A_tst, A_ref );
        params.error() = error;
        params.okay() = (error < tol);
    }
}

//------------------------------------------------------------------------------
void test_lasr( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_lasr_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_lasr_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_lasr_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_lasr_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
