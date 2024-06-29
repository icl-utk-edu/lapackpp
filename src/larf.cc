// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup reflector_aux_grp
void larf(
    lapack::Side side, int64_t m, int64_t n,
    float const* v, int64_t incv, float tau,
    float* C, int64_t ldc )
{
    char side_ = to_char( side );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int incv_ = to_lapack_int( incv );
    lapack_int ldc_ = to_lapack_int( ldc );

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    lapack::vector< float > work( lwork );

    LAPACK_slarf(
        &side_, &m_, &n_,
        v, &incv_, &tau,
        C, &ldc_,
        &work[0]
    );
}

// -----------------------------------------------------------------------------
/// @ingroup reflector_aux_grp
void larf(
    lapack::Side side, int64_t m, int64_t n,
    double const* v, int64_t incv, double tau,
    double* C, int64_t ldc )
{
    char side_ = to_char( side );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int incv_ = to_lapack_int( incv );
    lapack_int ldc_ = to_lapack_int( ldc );

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    lapack::vector< double > work( lwork );

    LAPACK_dlarf(
        &side_, &m_, &n_,
        v, &incv_, &tau,
        C, &ldc_,
        &work[0]
    );
}

// -----------------------------------------------------------------------------
/// @ingroup reflector_aux_grp
void larf(
    lapack::Side side, int64_t m, int64_t n,
    std::complex<float> const* v, int64_t incv, std::complex<float> tau,
    std::complex<float>* C, int64_t ldc )
{
    char side_ = to_char( side );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int incv_ = to_lapack_int( incv );
    lapack_int ldc_ = to_lapack_int( ldc );

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork );

    LAPACK_clarf(
        &side_, &m_, &n_,
        (lapack_complex_float*) v, &incv_, (lapack_complex_float*) &tau,
        (lapack_complex_float*) C, &ldc_,
        (lapack_complex_float*) &work[0]
    );
}

// -----------------------------------------------------------------------------
/// Applies a elementary reflector H to a m-by-n
/// matrix C, from either the left or the right. H is represented in the
/// form
/// \[
///     H = I - \tau v v^H,
/// \]
/// where $\tau$ is a scalar and v is a vector.
///
/// If $\tau = 0,$ then H is taken to be the unit matrix.
///
/// To apply $H^H,$ supply $\text{conj}(\tau)$ instead of $\tau.$
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] side
///     - lapack::Side::Left:  form $H C$
///     - lapack::Side::Right: form $C H$
///
/// @param[in] m
///     The number of rows of the matrix C.
///
/// @param[in] n
///     The number of columns of the matrix C.
///
/// @param[in] v
///     The vector v in the representation of H. v is not used if tau = 0.
///     - If side = Left,  the vector v of length 1 + (m-1)*abs(incv);
///     - if side = Right, the vector v of length 1 + (n-1)*abs(incv).
///
/// @param[in] incv
///     The increment between elements of v. incv != 0.
///
/// @param[in] tau
///     The value tau in the representation of H.
///
/// @param[in,out] C
///     The m-by-n matrix C, stored in an ldc-by-n array.
///     On entry, the m-by-n matrix C.
///     On exit, C is overwritten by the matrix $H C$ if side = Left,
///     or $C H$ if side = Right.
///
/// @param[in] ldc
///     The leading dimension of the array C. ldc >= max(1,m).
///
/// @ingroup reflector_aux_grp
void larf(
    lapack::Side side, int64_t m, int64_t n,
    std::complex<double> const* v, int64_t incv, std::complex<double> tau,
    std::complex<double>* C, int64_t ldc )
{
    char side_ = to_char( side );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int incv_ = to_lapack_int( incv );
    lapack_int ldc_ = to_lapack_int( ldc );

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork );

    LAPACK_zlarf(
        &side_, &m_, &n_,
        (lapack_complex_double*) v, &incv_, (lapack_complex_double*) &tau,
        (lapack_complex_double*) C, &ldc_,
        (lapack_complex_double*) &work[0]
    );
}

}  // namespace lapack
