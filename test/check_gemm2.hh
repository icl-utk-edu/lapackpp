// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// this is similar to blaspp/test/check_gemm.hh,
// except it uses LAPACK++ instead of calling Fortran LAPACK.

#ifndef CHECK_GEMM_HH
#define CHECK_GEMM_HH

#include "blas/util.hh"
#include "lapack.hh"

#include <limits>

// -----------------------------------------------------------------------------
// Computes error for multiplication with general matrix result.
// Covers dot, gemv, ger, geru, gemm, symv, hemv, symm, trmv, trsv?, trmm, trsm?.
// Cnorm is norm of original C, before multiplication operation.
template< typename T >
void check_gemm(
    int64_t m, int64_t n, int64_t k,
    T alpha,
    T beta,
    blas::real_type<T> Anorm,
    blas::real_type<T> Bnorm,
    blas::real_type<T> Cnorm,
    T const* Cref, int64_t ldcref,
    T* C, int64_t ldc,
    blas::real_type<T> error[1],
    int64_t* okay )
{
    #define    C(i_, j_)    C[ (i_) + (j_)*ldc ]
    #define Cref(i_, j_) Cref[ (i_) + (j_)*ldcref ]

    using real_t = blas::real_type<T>;

    require( m >= 0 );
    require( n >= 0 );
    require( k >= 0 );
    require( ldc >= m );
    require( ldcref >= m );

    // C -= Cref
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            C(i,j) -= Cref(i,j);
        }
    }

    error[0] = lapack::lange( lapack::Norm::Fro, m, n, C, ldc )
             / (sqrt(real_t(k)+2)*std::abs(alpha)*Anorm*Bnorm + 2*std::abs(beta)*Cnorm);

    // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
    real_t eps = std::numeric_limits< real_t >::epsilon();
    *okay = (error[0] < 3*eps);

    #undef C
    #undef Cref
}

// -----------------------------------------------------------------------------
// Computes error for multiplication with symmetric or Hermitian matrix result.
// Covers syr, syr2, syrk, syr2k, her, her2, herk, her2k.
// Cnorm is norm of original C, before multiplication operation.
//
// alpha and beta are either real or complex, depending on routine:
//          zher    zher2   zherk   zher2k  zsyr    zsyr2   zsyrk   zsyr2k
// alpha    real    complex real    complex complex complex complex complex
// beta     --      --      real    real    --      --      complex complex
// zsyr2 doesn't exist in standard BLAS or LAPACK.
template< typename TA, typename TB, typename T >
void check_herk(
    blas::Uplo uplo,
    int64_t n, int64_t k,
    TA alpha,
    TB beta,
    blas::real_type<T> Anorm,
    blas::real_type<T> Bnorm,
    blas::real_type<T> Cnorm,
    T const* Cref, int64_t ldcref,
    T* C, int64_t ldc,
    blas::real_type<T> error[1],
    int64_t* okay )
{
    #define    C(i_, j_)    C[ (i_) + (j_)*ldc ]
    #define Cref(i_, j_) Cref[ (i_) + (j_)*ldcref ]

    using real_t = blas::real_type<T>;

    require( n >= 0 );
    require( k >= 0 );
    require( ldc >= n );
    require( ldcref >= n );

    // C -= Cref
    if (uplo == blas::Uplo::Lower) {
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = j; i < n; ++i) {
                C(i,j) -= Cref(i,j);
            }
        }
    }
    else {
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i <= j; ++i) {
                C(i,j) -= Cref(i,j);
            }
        }
    }

    error[0] = lapack::lanhe( lapack::Norm::Fro, uplo, n, C, ldc )
             / (sqrt(real_t(k)+2)*std::abs(alpha)*Anorm*Bnorm + 2*std::abs(beta)*Cnorm);

    // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
    real_t eps = std::numeric_limits< real_t >::epsilon();
    *okay = (error[0] < 3*eps);

    #undef C
    #undef Cref
}

#endif        //  #ifndef CHECK_GEMM_HH
