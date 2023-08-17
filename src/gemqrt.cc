// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"

#if LAPACK_VERSION >= 30400  // >= 3.4.0

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gemqrt
int64_t gemqrt(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t k, int64_t nb,
    float const* V, int64_t ldv,
    float const* T, int64_t ldt,
    float* C, int64_t ldc )
{
    // for real, map ConjTrans to Trans
    if (trans == Op::ConjTrans)
        trans = Op::Trans;

    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int k_ = (lapack_int) k;
    lapack_int nb_ = (lapack_int) nb;
    lapack_int ldv_ = (lapack_int) ldv;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldc_ = (lapack_int) ldc;
    lapack_int info_ = 0;

    // Set workspace size
    lapack_int lwork_ = real((side == lapack::Side::Right) ? (m * nb) : (n * nb));

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_sgemqrt(
        &side_, &trans_, &m_, &n_, &k_, &nb_,
        V, &ldv_,
        T, &ldt_,
        C, &ldc_,
        &work[0], &info_ 
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
        );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gemqrt
int64_t gemqrt(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t k, int64_t nb,
    double const* V, int64_t ldv,
    double const* T, int64_t ldt,
    double* C, int64_t ldc )
{
    // for real, map ConjTrans to Trans
    if (trans == Op::ConjTrans)
        trans = Op::Trans;

    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int k_ = (lapack_int) k;
    lapack_int nb_ = (lapack_int) nb;
    lapack_int ldv_ = (lapack_int) ldv;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldc_ = (lapack_int) ldc;
    lapack_int info_ = 0;

    // Set workspace size
    lapack_int lwork_ = real((side == lapack::Side::Right) ? (m * nb) : (n * nb));

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dgemqrt(
        &side_, &trans_, &m_, &n_, &k_, &nb_,
        V, &ldv_,
        T, &ldt_,
        C, &ldc_,
        &work[0], &info_ 
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gemqrt
int64_t gemqrt(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t k, int64_t nb,
    std::complex<float> const* V, int64_t ldv,
    std::complex<float> const* T, int64_t ldt,
    std::complex<float>* C, int64_t ldc )
{
    // for complex, map Trans to ConjTrans
    if (trans == Op::Trans)
        trans = Op::ConjTrans;

    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int k_ = (lapack_int) k;
    lapack_int nb_ = (lapack_int) nb;
    lapack_int ldv_ = (lapack_int) ldv;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldc_ = (lapack_int) ldc;
    lapack_int info_ = 0;

    // Set workspace size
    lapack_int lwork_ = real((side == lapack::Side::Right) ? (m * nb) : (n * nb));

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );

    LAPACK_cgemqrt(
        &side_, &trans_, &m_, &n_, &k_, &nb_,
        (lapack_complex_float*) V, &ldv_,
        (lapack_complex_float*) T, &ldt_,
        (lapack_complex_float*) C, &ldc_,
        (lapack_complex_float*) &work[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Overwrites the general complex m-by-n matrix C with
///
/// - side = Left,  trans = NoTrans:   $Q C$
/// - side = Right, trans = NoTrans:   $C Q$
/// - side = Left,  trans = ConjTrans: $Q^H C$
/// - side = Right, trans = ConjTrans: $C Q^H$
///
/// where Q is a unitary matrix defined as the product of k
/// elementary reflectors:
///
///     Q = H(1) H(2) . . . H(k) = I - V T V^H
///
/// generated using the compact WY representation as returned by `lapack::geqrt`.
///
/// Q is of order m if side = Left and of order n if side = Right.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @since LAPACK 3.4.0
///
/// @param[in] side
///     - lapack::Side::Left: apply Q or Q^H from the Left;
///     - lapack::Side::Right: apply Q or Q^H from the Right.
///
/// @param[in] trans
///     - lapack::Op::NoTrans: No transpose, apply Q;
///     - lapack::Op::ConjTrans: Conjugate transpose, apply Q^H.
///
/// @param[in] m
///     The number of rows of the matrix C. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix C. n >= 0.
///
/// @param[in] k
///     The number of elementary reflectors whose product defines
///     the matrix Q.
///     If side = Left, m >= k >= 0;
///     if side = Right, n >= k >= 0.
///
/// @param[in] nb
///     The block size used for the storage of T. k >= nb >= 1.
///     This must be the same value of nb used to generate T
///     in `lapack::geqrt`.
///
/// @param[in] V
///     The ROWS-by-k matrix V, stored in an ldv-by-k array.
///     The i-th column must contain the vector which defines the
///     elementary reflector H(i), for i = 1,2,...,k, as returned by
///     `lapack::geqrt` in the first k columns of its array argument A.
///
/// @param[in] ldv
///     The leading dimension of the array V.
///     If side = Left, LDA >= max(1,m);
///     if side = Right, LDA >= max(1,n).
///
/// @param[in] T
///     The nb-by-k matrix T, stored in an ldt-by-k array.
///     The upper triangular factors of the block reflectors
///     as returned by `lapack::geqrt`, stored as a nb-by-n matrix.
///
/// @param[in] ldt
///     The leading dimension of the array T. ldt >= nb.
///
/// @param[in,out] C
///     The m-by-n matrix C, stored in an ldc-by-n array.
///     On entry, the m-by-n matrix C.
///     On exit, C is overwritten by Q C, Q^H C, C Q^H or C Q.
///
/// @param[in] ldc
///     The leading dimension of the array C. ldc >= max(1,m).
///
/// @retval = 0: successful exit
///
/// @ingroup gemqrt
int64_t gemqrt(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t k, int64_t nb,
    std::complex<double> const* V, int64_t ldv,
    std::complex<double> const* T, int64_t ldt,
    std::complex<double>* C, int64_t ldc )
{
    // for complex, map Trans to ConjTrans
    if (trans == Op::Trans)
        trans = Op::ConjTrans;

    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int k_ = (lapack_int) k;
    lapack_int nb_ = (lapack_int) nb;
    lapack_int ldv_ = (lapack_int) ldv;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldc_ = (lapack_int) ldc;
    lapack_int info_ = 0;

    // Set workspace size
    lapack_int lwork_ = real((side == lapack::Side::Right) ? (m * nb) : (n * nb));

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );

    LAPACK_zgemqrt(
        &side_, &trans_, &m_, &n_, &k_, &nb_,
        (lapack_complex_double*) V, &ldv_,
        (lapack_complex_double*) T, &ldt_,
        (lapack_complex_double*) C, &ldc_,
        (lapack_complex_double*) &work[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif    
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.4.0
