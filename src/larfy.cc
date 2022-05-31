// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#if LAPACK_VERSION >= 30700  // >= 3.7.0

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup unitary_computational
void larfy(
    lapack::Uplo uplo, int64_t n,
    float const* V, int64_t incv, float tau,
    float* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(incv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int incv_ = (lapack_int) incv;
    lapack_int ldc_ = (lapack_int) ldc;

    // allocate workspace
    lapack::vector< float > work( (n) );

    LAPACK_slarfy(
        &uplo_, &n_,
        V, &incv_, &tau,
        C, &ldc_,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// @ingroup unitary_computational
void larfy(
    lapack::Uplo uplo, int64_t n,
    double const* V, int64_t incv, double tau,
    double* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(incv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int incv_ = (lapack_int) incv;
    lapack_int ldc_ = (lapack_int) ldc;

    // allocate workspace
    lapack::vector< double > work( (n) );

    LAPACK_dlarfy(
        &uplo_, &n_,
        V, &incv_, &tau,
        C, &ldc_,
        &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// @ingroup unitary_computational
void larfy(
    lapack::Uplo uplo, int64_t n,
    std::complex<float> const* V, int64_t incv, std::complex<float> tau,
    std::complex<float>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(incv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int incv_ = (lapack_int) incv;
    lapack_int ldc_ = (lapack_int) ldc;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (n) );

    LAPACK_clarfy(
        &uplo_, &n_,
        (lapack_complex_float*) V, &incv_, (lapack_complex_float*) &tau,
        (lapack_complex_float*) C, &ldc_,
        (lapack_complex_float*) &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// Applies an elementary reflector, or Householder matrix, H,
/// to an n x n Hermitian matrix C, from both the left and the right.
///
/// H is represented in the form
/// \[
///     H = I - \tau v v^H
/// \]
/// where $\tau$ is a scalar and $v$ is a vector.
///
/// If $tau$ is zero, then $H$ is taken to be the unit matrix.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @since LAPACK 3.7.0
///
/// @param[in] uplo
///     Whether the upper or lower triangular part of the
///     Hermitian matrix C is stored.
///     - lapack::Uplo::Upper: Upper triangle
///     - lapack::Uplo::Lower: Lower triangle
///
/// @param[in] n
///     The number of rows and columns of the matrix C. n >= 0.
///
/// @param[in] V
///     The vector V of length 1 + (n-1)*abs(incv).
///
/// @param[in] incv
///     The increment between successive elements of v. incv must
///     not be zero.
///
/// @param[in] tau
///     The value tau as described above.
///
/// @param[in,out] C
///     The n-by-n matrix C, stored in an ldc-by-n array.
///     On entry, the matrix C.
///     On exit, C is overwritten by $H C H^H$.
///
/// @param[in] ldc
///     The leading dimension of the array C. ldc >= max( 1, n ).
///
/// @ingroup unitary_computational
void larfy(
    lapack::Uplo uplo, int64_t n,
    std::complex<double> const* V, int64_t incv, std::complex<double> tau,
    std::complex<double>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(incv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int incv_ = (lapack_int) incv;
    lapack_int ldc_ = (lapack_int) ldc;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (n) );

    LAPACK_zlarfy(
        &uplo_, &n_,
        (lapack_complex_double*) V, &incv_, (lapack_complex_double*) &tau,
        (lapack_complex_double*) C, &ldc_,
        (lapack_complex_double*) &work[0]
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
}

}  // namespace lapack

#endif  // LAPACK >= 3.7.0
