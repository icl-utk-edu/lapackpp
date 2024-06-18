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
void test_lae2_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;
    using blas::conj;
    using lapack::Job, lapack::Uplo;

    // Constants
    const real_t eps = std::numeric_limits< real_t >::epsilon();

    // get & mark input values
    params.dim.m() = 2;
    params.dim.n() = 2;
    real_t tol = params.tol() * eps;
    int verbose = params.verbose();
    params.matrix.mark();

    // mark non-standard output values
    params.error.name( "Lambda" );

    if (! run)
        return;

    //---------- setup
    int64_t n = 2;
    int64_t lda = 2;
    std::vector< scalar_t > A( lda*n );

    lapack::generate_matrix( params.matrix, n, n, &A[0], lda );

    // A = [ a        b ], stored column-wise.
    //     [ conj(b)  c ]
    scalar_t a = A[ 0 ];
    scalar_t b = A[ 2 ];
    scalar_t c = A[ 3 ];
    A[ 1 ] = conj( b );

    real_t rt1, rt2, rt1_ref, rt2_ref, cs1;
    scalar_t sn1;

    if (verbose >= 2) {
        printf( "A = " ); print_matrix( n, n, &A[0], lda );
    }

    //---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    // no info returned
    lapack::lae2( a, b, c, &rt1, &rt2 );
    time = testsweeper::get_wtime() - time;

    params.time() = time;

    std::vector< real_t > Lambda{ rt1, rt2 };

    if (verbose >= 2) {
        printf( "Lambda = " ); print_vector( n, &Lambda[0], 1 );
    }

    if (params.check() == 'y') {
        //---------- run reference, using laev2
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        lapack::laev2( a, b, c, &rt1_ref, &rt2_ref, &cs1, &sn1 );
        time = testsweeper::get_wtime() - time;

        params.ref_time() = time;

        //---------- check error compared to reference
        std::vector< real_t > Lambda_ref{ rt1_ref, rt2_ref };
        real_t error = rel_error( Lambda, Lambda_ref );
        params.error() = error;
        params.okay() = (error < tol);
    }
}

//------------------------------------------------------------------------------
void test_lae2( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_lae2_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_lae2_work< double >( params, run );
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
