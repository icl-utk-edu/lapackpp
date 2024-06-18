// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "lapack.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "check_heev.hh"
#include "lapacke_wrappers.hh"

#include <vector>

//------------------------------------------------------------------------------
template< typename scalar_t >
void test_laev2_work( Params& params, bool run )
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
    params.ortho();
    params.error2();
    params.error2.name( "Lambda" );

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

    real_t rt1, rt2, cs1;
    scalar_t sn1;

    if (verbose >= 2) {
        printf( "A = " ); print_matrix( n, n, &A[0], lda );
    }

    //---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    // no info returned
    lapack::laev2( a, b, c, &rt1, &rt2, &cs1, &sn1 );
    time = testsweeper::get_wtime() - time;

    params.time() = time;

    // Z = [ cs1  -conj( sn1 ) ], stored column-wise.
    //     [ sn1  cs1          ]
    std::vector< scalar_t > Z{ cs1, sn1, -conj( sn1 ), cs1 };
    std::vector< real_t > Lambda{ rt1, rt2 };

    if (verbose >= 2) {
        printf( "Z = " ); print_matrix( n, n, &Z[0], lda );
        printf( "Lambda = " ); print_vector( n, &Lambda[0], 1 );
    }

    if (params.check() == 'y') {
        int64_t ldz = 2;

        // ---------- check numerical error
        // result[ 0 ] = || A - Z Lambda Z^H || / (n ||A||), if jobz != NoVec.
        // result[ 1 ] = || I - Z^H Z || / n, if jobz != NoVec.
        // result[ 2 ] = 0 if Lambda is in non-decreasing order, else >= 1.
        // Ignored; laev2 returns rt1 >= rt2.
        real_t result[ 3 ] = { (real_t) testsweeper::no_data_flag,
                               (real_t) testsweeper::no_data_flag,
                               (real_t) testsweeper::no_data_flag };

        check_heev( Job::Vec, Uplo::Upper, n, &A[0], lda,
                    n, &Lambda[0], &Z[0], ldz, result );

        // 1 (true) if rt1 < rt2, 0 (false) if rt1 >= rt2.
        result[ 2 ] = (rt1 < rt2);

        params.error()  = result[ 0 ];
        params.ortho()  = result[ 1 ];
        params.error2() = result[ 2 ];
        params.okay()   = result[ 0 ] < tol
                       && result[ 1 ] < tol
                       && result[ 2 ] < tol;
    }
}

//------------------------------------------------------------------------------
void test_laev2( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_laev2_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_laev2_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_laev2_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_laev2_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
