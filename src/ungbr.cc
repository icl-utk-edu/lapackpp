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
/// @ingroup gesvd_computational
int64_t ungbr(
    lapack::Vect vect, int64_t m, int64_t n, int64_t k,
    std::complex<float>* A, int64_t lda,
    std::complex<float> const* tau )
{
    char vect_ = to_char( vect );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int k_ = to_lapack_int( k );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_cungbr(
        &vect_, &m_, &n_, &k_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) qry_work, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );

    LAPACK_cungbr(
        &vect_, &m_, &n_, &k_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) &work[0], &lwork_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Generates one of the complex unitary matrices $Q$ or $P^H$
/// determined by `lapack::gebrd` when reducing a complex matrix A to bidiagonal
/// form: $A = Q B P^H.$ $Q$ and $P^H$ are defined as products of
/// elementary reflectors H(i) or G(i) respectively.
///
/// - If vect = Q, A is assumed to have been an m-by-k matrix,
///   and Q is of order m:
///   - if m >= k, $Q = H(1) H(2) \dots H(k)$
///     and `ungbr` returns the first n columns of Q, where m >= n >= k;
///   - if m < k, $Q = H(1) H(2) \dots H(m-1)$
///     and `ungbr` returns Q as an m-by-m matrix.
///
/// - If vect = P, A is assumed to have been a k-by-n matrix,
///   and $P^H$ is of order n:
///   - if k < n, $P^H = G(k) \dots G(2) G(1)$
///     and `ungbr` returns the first m rows of $P^H,$ where n >= m >= k;
///   - if k >= n, $P^H = G(n-1) \dots G(2) G(1)$
///     and `ungbr` returns $P^H$ as an n-by-n matrix.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this is an alias for `lapack::orgbr`.
///
/// @param[in] vect
///     Specifies whether the matrix $Q$ or the matrix $P^H$ is
///     required, as defined in the transformation applied by `lapack::gebrd`:
///     - lapack::Vect::Q: generate $Q;$
///     - lapack::Vect::P: generate $P^H,$
///
/// @param[in] m
///     The number of rows of the matrix Q or $P^H$ to be returned.
///     m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix Q or $P^H$ to be returned.
///     n >= 0.
///     - If vect = Q, m >= n >= min(m,k);
///     - if vect = P, n >= m >= min(n,k).
///
/// @param[in] k
///     - If vect = Q, the number of columns in the original m-by-k
///     matrix reduced by `lapack::gebrd`.
///     - If vect = P, the number of rows in the original k-by-n
///     matrix reduced by `lapack::gebrd`.
///     - k >= 0.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     On entry, the vectors which define the elementary reflectors,
///     as returned by `lapack::gebrd`.
///     On exit, the m-by-n matrix $Q$ or $P^H,$
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= m.
///
/// @param[in] tau
///     tau(i) must contain the scalar factor of the elementary
///     reflector H(i) or G(i), which determines $Q$ or $P^H,$ as
///     returned by `lapack::gebrd` in its array argument tauq or taup.
///     - If vect = Q, the vector tau of length min(m,k);
///     - if vect = P, the vector tau of length min(n,k).
///
/// @return = 0: successful exit
///
/// @ingroup gesvd_computational
int64_t ungbr(
    lapack::Vect vect, int64_t m, int64_t n, int64_t k,
    std::complex<double>* A, int64_t lda,
    std::complex<double> const* tau )
{
    char vect_ = to_char( vect );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int k_ = to_lapack_int( k );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_zungbr(
        &vect_, &m_, &n_, &k_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) qry_work, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );

    LAPACK_zungbr(
        &vect_, &m_, &n_, &k_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) &work[0], &lwork_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
