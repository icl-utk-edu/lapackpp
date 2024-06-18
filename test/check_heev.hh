// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas.hh"
#include "lapack.hh"
#include "error.hh"
#include "check_ortho.hh"
#include "scale.hh"

#include <vector>

//------------------------------------------------------------------------------
// Computes error measures:
// If jobz != NoVec:
//     result[ 0 ] = || A - Z Lambda Z^H || / (n ||A||) if nfound == n;
//     result[ 0 ] = || Z^H A Z - Lambda || / (n ||A||) otherwise.
//     result[ 1 ] = || I - Z^H Z || / n, if jobz != NoVec.
// result[ 2 ] = 0 if Lambda is in non-decreasing order, else >= 1.
template< typename scalar_t >
void check_heev(
    lapack::Job jobz,
    lapack::Uplo uplo, int64_t n,
    scalar_t const* A, int64_t lda,
    int64_t nfound,
    blas::real_type< scalar_t > const* Lambda,
    scalar_t const* Z, int64_t ldz,
    blas::real_type< scalar_t > result[ 3 ] )
{
    using namespace blas;
    using namespace lapack;
    using real_t = blas::real_type< scalar_t >;

    // Constants
    const scalar_t one  = 1;
    const scalar_t zero = 0;

    if (jobz == Job::Vec) {
        real_t Anorm = lapack::lanhe( Norm::One, uplo, n, A, lda );

        // R is nfound-by-nfound, whether n == nfound or not.
        int64_t ldr = nfound;
        std::vector< scalar_t > R_( ldr*nfound );
        scalar_t* R = &R_[ 0 ];

        if (n == nfound) {
            // || A - Z Lambda Z^H ||
            std::vector< scalar_t > ZLambda_( ldz*n );
            scalar_t* ZLambda = &ZLambda_[ 0 ];

            // ZLambda = Z Lambda is n-by-n.
            lapack::lacpy( MatrixType::General, n, n,
                           Z, ldz,
                           ZLambda, ldz );
            col_scale( n, n, ZLambda, ldz, Lambda );
            // R = A - (Z Lambda) Z^H; could use gemmtr instead of gemm.
            lapack::lacpy( MatrixType::General, n, n,
                           A, lda,
                           R, ldr );
            blas::gemm( Layout::ColMajor, Op::NoTrans, Op::ConjTrans, n, n, n,
                        -one, ZLambda, ldz,
                              Z, ldz,
                        one,  R, ldr );
        }
        else {
            // || Z^H A Z - Lambda ||
            std::vector< scalar_t > AZ_( lda*nfound );
            scalar_t* AZ = &AZ_[ 0 ];

            // AZ = A Z is n-by-nfound.
            blas::hemm( Layout::ColMajor, Side::Left, uplo, n, nfound,
                        one,  A, lda,
                              Z, ldz,
                        zero, AZ, lda );
            // R = Z^H (A Z); could use gemmtr instead of gemm.
            blas::gemm( Layout::ColMajor, Op::ConjTrans, Op::NoTrans,
                        nfound, nfound, n,
                        one,  Z, ldz,
                              AZ, lda,
                        zero, R, ldr );
            // R -= Lambda, along diagonal.
            blas::axpy( nfound, -one, Lambda, 1, R, ldr + 1 );
        }
        result[ 0 ] = lapack::lanhe( Norm::One, uplo, nfound, R, ldr )
                    / (n * Anorm);

        result[ 1 ] = check_orthogonality( RowCol::Col, n, nfound, Z, ldz );
    }

    // Check that Lambda is non-decreasing.
    result[ 2 ] = 0;
    for (int64_t i = 0; i < nfound - 1; ++i) {
        if (Lambda[ i ] > Lambda[ i+1 ])
            result[ 2 ] += 1;
    }
}
