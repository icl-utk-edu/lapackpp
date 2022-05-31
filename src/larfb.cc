// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup unitary_computational
void larfb(
    lapack::Side side, lapack::Op trans,
    lapack::Direction direction, lapack::StoreV storev,
    int64_t m, int64_t n, int64_t k,
    float const* V, int64_t ldv,
    float const* T, int64_t ldt,
    float* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    char direction_ = direction2char( direction );
    char storev_ = storev2char( storev );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int k_ = (lapack_int) k;
    lapack_int ldv_ = (lapack_int) ldv;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldc_ = (lapack_int) ldc;

    // from docs
    lapack_int ldwork_ = (side == Side::Left ? n : m);

    // allocate workspace
    lapack::vector< float > work( ldwork_ * k );

    LAPACK_slarfb(
        &side_, &trans_, &direction_, &storev_, &m_, &n_, &k_,
        V, &ldv_,
        T, &ldt_,
        C, &ldc_,
        &work[0], &ldwork_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1, 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// @ingroup unitary_computational
void larfb(
    lapack::Side side, lapack::Op trans, lapack::Direction direction, lapack::StoreV storev,
    int64_t m, int64_t n, int64_t k,
    double const* V, int64_t ldv,
    double const* T, int64_t ldt,
    double* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    char direction_ = direction2char( direction );
    char storev_ = storev2char( storev );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int k_ = (lapack_int) k;
    lapack_int ldv_ = (lapack_int) ldv;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldc_ = (lapack_int) ldc;

    // from docs
    lapack_int ldwork_ = (side == Side::Left ? n : m);

    // allocate workspace
    lapack::vector< double > work( ldwork_ * k );

    LAPACK_dlarfb(
        &side_, &trans_, &direction_, &storev_, &m_, &n_, &k_,
        V, &ldv_,
        T, &ldt_,
        C, &ldc_,
        &work[0], &ldwork_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1, 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// @ingroup unitary_computational
void larfb(
    lapack::Side side, lapack::Op trans,
    lapack::Direction direction, lapack::StoreV storev,
    int64_t m, int64_t n, int64_t k,
    std::complex<float> const* V, int64_t ldv,
    std::complex<float> const* T, int64_t ldt,
    std::complex<float>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    char direction_ = direction2char( direction );
    char storev_ = storev2char( storev );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int k_ = (lapack_int) k;
    lapack_int ldv_ = (lapack_int) ldv;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldc_ = (lapack_int) ldc;

    // from docs
    lapack_int ldwork_ = (side == Side::Left ? n : m);

    // allocate workspace
    lapack::vector< std::complex<float> > work( ldwork_ * k );

    LAPACK_clarfb(
        &side_, &trans_, &direction_, &storev_, &m_, &n_, &k_,
        (lapack_complex_float*) V, &ldv_,
        (lapack_complex_float*) T, &ldt_,
        (lapack_complex_float*) C, &ldc_,
        (lapack_complex_float*) &work[0], &ldwork_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1, 1
        #endif
    );
}

// -----------------------------------------------------------------------------
/// Applies a block reflector $H$ or its transpose $H^H$ to a
/// m-by-n matrix C, from either the left or the right.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] side
///     - lapack::Side::Left:  apply $H$ or $H^H$ from the Left
///     - lapack::Side::Right: apply $H$ or $H^H$ from the Right
///
/// @param[in] trans
///     - lapack::Op::NoTrans:   apply $H  $ (No transpose)
///     - lapack::Op::ConjTrans: apply $H^H$ (Conjugate transpose)
///
/// @param[in] direction
///     Indicates how H is formed from a product of elementary
///     reflectors
///     - lapack::Direction::Forward:  $H = H(1) H(2) \dots H(k)$
///     - lapack::Direction::Backward: $H = H(k) \dots H(2) H(1)$
///
/// @param[in] storev
///     Indicates how the vectors which define the elementary
///     reflectors are stored:
///     - lapack::StoreV::Columnwise
///     - lapack::StoreV::Rowwise
///
/// @param[in] m
///     The number of rows of the matrix C.
///
/// @param[in] n
///     The number of columns of the matrix C.
///
/// @param[in] k
///     The order of the matrix T (= the number of elementary
///     reflectors whose product defines the block reflector).
///     - If side = Left,  m >= k >= 0;
///     - if side = Right, n >= k >= 0.
///
/// @param[in] V
///     - If storev = Columnwise:
///       - if side = Left,  the m-by-k matrix V, stored in an ldv-by-k array;
///       - if side = Right, the n-by-k matrix V, stored in an ldv-by-k array.
///     - If storev = Rowwise:
///       - if side = Left,  the k-by-m matrix V, stored in an ldv-by-m array;
///       - if side = Right, the k-by-n matrix V, stored in an ldv-by-n array.
///     - See Further Details.
///
/// @param[in] ldv
///     The leading dimension of the array V.
///     - If storev = Columnwise and side = Left,  ldv >= max(1,m);
///     - if storev = Columnwise and side = Right, ldv >= max(1,n);
///     - if storev = Rowwise, ldv >= k.
///
/// @param[in] T
///     The k-by-k matrix T, stored in an ldt-by-k array.
///     The triangular k-by-k matrix T in the representation of the
///     block reflector.
///
/// @param[in] ldt
///     The leading dimension of the array T. ldt >= k.
///
/// @param[in,out] C
///     The m-by-n matrix C, stored in an ldc-by-n array.
///     On entry, the m-by-n matrix C.
///     On exit, C is overwritten by
///     $H C$ or $H^H C$ or $C H$ or $C H^H$.
///
/// @param[in] ldc
///     The leading dimension of the array C. ldc >= max(1,m).
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The shape of the matrix V and the storage of the vectors which define
/// the H(i) is best illustrated by the following example with n = 5 and
/// k = 3. The elements equal to 1 are not stored. The rest of the
/// array is not used.
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
void larfb(
    lapack::Side side, lapack::Op trans,
    lapack::Direction direction, lapack::StoreV storev,
    int64_t m, int64_t n, int64_t k,
    std::complex<double> const* V, int64_t ldv,
    std::complex<double> const* T, int64_t ldt,
    std::complex<double>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    char direction_ = direction2char( direction );
    char storev_ = storev2char( storev );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int k_ = (lapack_int) k;
    lapack_int ldv_ = (lapack_int) ldv;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldc_ = (lapack_int) ldc;

    // from docs
    lapack_int ldwork_ = (side == Side::Left ? n : m);

    // allocate workspace
    lapack::vector< std::complex<double> > work( ldwork_ * k );

    LAPACK_zlarfb(
        &side_, &trans_, &direction_, &storev_, &m_, &n_, &k_,
        (lapack_complex_double*) V, &ldv_,
        (lapack_complex_double*) T, &ldt_,
        (lapack_complex_double*) C, &ldc_,
        (lapack_complex_double*) &work[0], &ldwork_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1, 1
        #endif
    );
}

}  // namespace lapack
