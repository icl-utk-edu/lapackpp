// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup unitary_computational
void larft(
    lapack::Direction direction, lapack::StoreV storev,
    int64_t n, int64_t k,
    float const* V, int64_t ldv,
    float const* tau,
    float* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
    }
    char direction_ = direction2char( direction );
    char storev_ = storev2char( storev );
    lapack_int n_ = (lapack_int) n;
    lapack_int k_ = (lapack_int) k;
    lapack_int ldv_ = (lapack_int) ldv;
    lapack_int ldt_ = (lapack_int) ldt;

    LAPACK_slarft(
        &direction_, &storev_, &n_, &k_,
        V, &ldv_,
        tau,
        T, &ldt_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// @ingroup unitary_computational
void larft(
    lapack::Direction direction, lapack::StoreV storev,
    int64_t n, int64_t k,
    double const* V, int64_t ldv,
    double const* tau,
    double* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
    }
    char direction_ = direction2char( direction );
    char storev_ = storev2char( storev );
    lapack_int n_ = (lapack_int) n;
    lapack_int k_ = (lapack_int) k;
    lapack_int ldv_ = (lapack_int) ldv;
    lapack_int ldt_ = (lapack_int) ldt;

    LAPACK_dlarft(
        &direction_, &storev_, &n_, &k_,
        V, &ldv_,
        tau,
        T, &ldt_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// @ingroup unitary_computational
void larft(
    lapack::Direction direction, lapack::StoreV storev,
    int64_t n, int64_t k,
    std::complex<float> const* V, int64_t ldv,
    std::complex<float> const* tau,
    std::complex<float>* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
    }
    char direction_ = direction2char( direction );
    char storev_ = storev2char( storev );
    lapack_int n_ = (lapack_int) n;
    lapack_int k_ = (lapack_int) k;
    lapack_int ldv_ = (lapack_int) ldv;
    lapack_int ldt_ = (lapack_int) ldt;

    LAPACK_clarft(
        &direction_, &storev_, &n_, &k_,
        (lapack_complex_float*) V, &ldv_,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) T, &ldt_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// Forms the triangular factor T of a complex block reflector H
/// of order n, which is defined as a product of k elementary reflectors.
///
/// If direction = Forward, $H = H(1) H(2) \dots H(k)$ and T is upper triangular;
///
/// If direction = Backward, $H = H(k) \dots H(2) H(1)$ and T is lower triangular.
///
/// If storev = Columnwise, the vector which defines the elementary reflector
/// H(i) is stored in the i-th column of the array V, and
/// \[
///     H = I - V T V^H.
/// \]
/// If storev = Rowwise, the vector which defines the elementary reflector
/// H(i) is stored in the i-th row of the array V, and
/// \[
///     H = I - V^H T V.
/// \]
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] direction
///     Specifies the order in which the elementary reflectors are
///     multiplied to form the block reflector:
///     - lapack::Direction::Forward:  $H = H(1) H(2) \dots H(k)$
///     - lapack::Direction::Backward: $H = H(k) \dots H(2) H(1)$
///
/// @param[in] storev
///     Specifies how the vectors which define the elementary
///     reflectors are stored (see also Further Details):
///     - lapack::StoreV::Columnwise
///     - lapack::StoreV::Rowwise
///
/// @param[in] n
///     The order of the block reflector H. n >= 0.
///
/// @param[in] k
///     The order of the triangular factor T (= the number of
///     elementary reflectors). k >= 1.
///
/// @param[in] V
///     - If storev = Columnwise, the n-by-k matrix V, stored in an ldv-by-k array;
///     - if storev = Rowwise,    the k-by-n matrix V, stored in an ldv-by-n array.
///     \n
///     See further details.
///
/// @param[in] ldv
///     The leading dimension of the array V.
///     - If storev = Columnwise, ldv >= max(1,n);
///     - if storev = Rowwise,    ldv >= k.
///
/// @param[in] tau
///     The vector tau of length k.
///     tau(i) must contain the scalar factor of the elementary
///     reflector H(i).
///
/// @param[out] T
///     The k-by-k matrix T, stored in an ldt-by-k array.
///     The k-by-k triangular factor T of the block reflector.
///     - If direction = Forward, T is upper triangular;
///     - if direction = Backward, T is lower triangular.
///     \n
///     The rest of the array is not used.
///
/// @param[in] ldt
///     The leading dimension of the array T. ldt >= k.
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The shape of the matrix V and the storage of the vectors which define
/// the H(i) is best illustrated by the following example with n = 5 and
/// k = 3. The elements equal to 1 are not stored.
///
///     direction = Forward and          direction = Forward and
///     storev = Columnwise:             storev = Rowwise:
///
///     V = (  1       )                 V = (  1 v1 v1 v1 v1 )
///         ( v1  1    )                     (     1 v2 v2 v2 )
///         ( v1 v2  1 )                     (        1 v3 v3 )
///         ( v1 v2 v3 )
///         ( v1 v2 v3 )
///
///     direction = Backward and         direction = Backward and
///     storev = Columnwise:             storev = Rowwise:
///
///     V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
///         ( v1 v2 v3 )                     ( v2 v2 v2  1    )
///         (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
///         (     1 v3 )
///         (        1 )
///
/// @ingroup unitary_computational
void larft(
    lapack::Direction direction, lapack::StoreV storev,
    int64_t n, int64_t k,
    std::complex<double> const* V, int64_t ldv,
    std::complex<double> const* tau,
    std::complex<double>* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
    }
    char direction_ = direction2char( direction );
    char storev_ = storev2char( storev );
    lapack_int n_ = (lapack_int) n;
    lapack_int k_ = (lapack_int) k;
    lapack_int ldv_ = (lapack_int) ldv;
    lapack_int ldt_ = (lapack_int) ldt;

    LAPACK_zlarft(
        &direction_, &storev_, &n_, &k_,
        (lapack_complex_double*) V, &ldv_,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) T, &ldt_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
}

}  // namespace lapack
