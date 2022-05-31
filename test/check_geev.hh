// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas.hh"
#include "lapack.hh"

#include <vector>

// -----------------------------------------------------------------------------
// work = V*W^{trans}; used in check_geev
// version for real
template< typename real_t >
void check_geev_multiply_VW(
    blas::Op trans, int64_t n,
    std::complex<real_t> const* W,
    real_t const* V, int64_t ldv,
    real_t* work, int64_t ldwork )
{
    #define    V(i_, j_)    V[ (i_) + (j_)*ldv ]
    #define work(i_, j_) work[ (i_) + (j_)*ldwork ]

    // W is block diagonal, with 2x2 blocks for each complex eigenvalue pair.
    lapack::laset( lapack::MatrixType::General, n, n, 0., 0., &work(0,0), n );
    bool ipair = false;
    for (int64_t j = 0; j < n; ++j) {
        if (ipair) {
            // 2nd part of complex eigenvector, already handled
            ipair = false;
        }
        else if (imag( W[j] ) != 0 && j+1 < n) {
            // complex eigenvector
            ipair = true;
            // Wmat = [  Wr  Wi ] (stored columnwise)
            //        [ -Wi  Wr ]
            real_t Wmat[4] = { real( W[j] ), -imag( W[j] ),
                               imag( W[j] ),  real( W[j] ) };
            // work(:,j:j+1) = V(:,j:j+1) * Wmat^{trans}
            blas::gemm( blas::Layout::ColMajor,
                        blas::Op::NoTrans, trans, n, 2, 2,
                        1.0, &V(0,j), ldv,
                             Wmat, 2,
                        0.0, &work(0,j), n );
        }
        else {
            // work(:,j) = V(:,j) * W[j]
            blas::axpy( n, real( W[j] ), &V(0,j), 1, &work(0,j), 1 );
        }
    }

    #undef V
    #undef work
}

// -----------------------------------------------------------------------------
// work = V*W; used in check_geev
// version for complex
template< typename real_t >
void check_geev_multiply_VW(
    blas::Op trans, int64_t n,
    std::complex<real_t> const* W,
    std::complex<real_t> const* V, int64_t ldv,
    std::complex<real_t>* work, int64_t ldwork )
{
    #define    V(i_, j_)    V[ (i_) + (j_)*ldv ]
    #define work(i_, j_) work[ (i_) + (j_)*ldwork ]

    // W is diagonal.
    lapack::laset( lapack::MatrixType::General, n, n, 0., 0., &work(0,0), n );
    for (int64_t j = 0; j < n; ++j) {
        std::complex<real_t> Wtmp;
        if (trans == blas::Op::ConjTrans)
            Wtmp = conj( W[j] );
        else
            Wtmp = W[j];

        // work(:,j) = V(:,j) * W[j]
        blas::axpy( n, Wtmp, &V(0,j), 1, &work(0,j), 1 );
    }

    #undef V
    #undef work
}

// -----------------------------------------------------------------------------
// checks | ||V||_2 - 1 |
// 2-norm as used in drvev, not max-norm as used in get22
// also checks that conjugate pairs are together correctly
// version for real
template< typename real_t >
real_t check_geev_Vnormalization(
    int64_t n,
    std::complex<real_t> const* W,
    real_t const* V, int64_t ldv )
{
    using namespace blas;

    #define V(i_, j_) V[ (i_) + (j_)*ldv ]

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t error = 0;
    bool ipair = false;
    for (int64_t j = 0; j < n; ++j) {
        if (imag( W[j] ) < 0) {
            if (ipair != true) {
                error = std::numeric_limits< real_t >::infinity();
            }
            ipair = false;
        }
        else {
            if (ipair == true) {
                error = std::numeric_limits< real_t >::infinity();
            }
            real_t nrm;
            if (imag( W[j] ) > 0) {
                // complex eigenvector
                require( j < n-1 );
                require( real( W[j] ) ==  real( W[j+1] ) );
                require( imag( W[j] ) == -imag( W[j+1] ) );
                ipair = true;
                nrm = lapack::lapy2( blas::nrm2( n, &V(0,j  ), 1 ),
                                     blas::nrm2( n, &V(0,j+1), 1 ) );
            }
            else {
                // real eigenvector
                nrm = blas::nrm2( n, &V(0,j), 1 );
            }
            error = max( error, std::abs( nrm - 1 ) );

            // check largest component is real; set to inf if not
            if (imag( W[j] ) > 0) {
                real_t Vmax = 0;
                real_t Vrmax = 0;
                for (int64_t i = 0; i < n; ++i) {
                    real_t tmp = lapack::lapy2( V(i,j), V(i,j+1) );
                    if (tmp > Vmax)
                        Vmax = tmp;
                    if (V(i,j+1) == 0 && std::abs( V(i,j) ) > Vrmax)
                        Vrmax = std::abs( V(i,j) );
                }
                if (Vrmax / Vmax < 1 - 2*eps) {
                    //printf( "Vrmax %.2e, Vmax %.2e\n", Vrmax, Vmax );
                    error = std::numeric_limits< real_t >::infinity();
                }
            }
        }
    }
    return error;

    #undef V
}

// -----------------------------------------------------------------------------
// checks | ||V||_2 - 1 |
// 2-norm as used in drvev, not max-norm as used in get22
// also checks that conjugate pairs are together correctly
// version for complex
template< typename real_t >
real_t check_geev_Vnormalization(
    int64_t n,
    std::complex<real_t> const* W,
    std::complex<real_t> const* V, int64_t ldv )
{
    using namespace blas;

    #define V(i_, j_) V[ (i_) + (j_)*ldv ]

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t error = 0;
    for (int64_t j = 0; j < n; ++j) {
        real_t nrm = blas::nrm2( n, &V(0,j), 1 );
        error = max( error, std::abs( nrm - 1 ) );

        // check largest component is real; set to inf if not
        if (imag( W[j] ) > 0) {
            real_t Vmax = 0;
            real_t Vrmax = 0;
            for (int64_t i = 0; i < n; ++i) {
                real_t tmp = std::abs( V(i,j) );
                if (tmp > Vmax)
                    Vmax = tmp;
                if (imag( V(i,j) ) == 0 && std::abs( real( V(i,j) )) > Vrmax)
                    Vrmax = std::abs( real( V(i,j) ));
            }
            if (Vrmax / Vmax < 1 - 2*eps) {
                //printf( "Vrmax %.2e, Vmax %.2e\n", Vrmax, Vmax );
                error = std::numeric_limits< real_t >::infinity();
            }
        }
    }
    return error;

    #undef V
}

// -----------------------------------------------------------------------------
// || op(A) V - V W ||_1 / (||A||_1 ||V||_1)
// max_j | || E(j) ||_2 - 1 | / n
//
// For right eigenvectors, use trans = NoTrans
// to compute || A VR - VR W || / (||A|| ||VR||)
//
// For left eigenvectors, use trans = ConjTrans
// to compute || A^H VL - VL W^H || / (||A|| ||V||)
//
// TODO: doesn't quite match LAWN 41, which normalizes by (n ||A||)
// See LAPACK testing drvev and get22.

// version for real
template< typename scalar_t >
void check_geev(
    blas::Op trans, int64_t n,
    scalar_t const* A, int64_t lda,
    blas::complex_type< scalar_t > const* W,
    scalar_t const* V, int64_t ldv,
    int64_t verbose,
    blas::real_type< scalar_t > results[2] )
{
    using real_t = blas::real_type< scalar_t >;

    // work = op(A) V - work = op(A) V - (V W)
    std::vector< scalar_t > work( n * n );
    check_geev_multiply_VW( trans, n, W, V, ldv, &work[0], n );

    if (verbose >= 2) {
        printf( "VW = " ); print_matrix( n, n, &work[0], n );
    }

    blas::gemm( blas::Layout::ColMajor,
                trans, blas::Op::NoTrans, n, n, n,
                 1.0, A, lda,
                      V, ldv,
                -1.0, &work[0], n );

    if (verbose >= 2) {
        printf( "R = " ); print_matrix( n, n, &work[0], n );
    }

    // || op(A) V - V W || / (n ||A||)
    // instead of n, get22 uses ||V||, which generally gives a larger error:
    // || op(A) V - V W || / (||V|| ||A||)
    real_t error = lapack::lange( lapack::Norm::One, n, n, &work[0], n );
    real_t Anorm = lapack::lange( lapack::Norm::One, n, n, A, lda );
    real_t Vnorm = lapack::lange( lapack::Norm::One, n, n, V, ldv );
    results[0] = error / Vnorm / Anorm;

    if (verbose >= 1) {
        printf( "error: { ||A^{%c} V - V W||=%.2e / (||V||=%.2e ||A||=%.2e) } = %.2e;  n=%.0f\n",
                op2char(trans), error, Vnorm, Anorm, results[0], real_t(n) );
    }

    results[1] = check_geev_Vnormalization( n, W, V, ldv );
}
