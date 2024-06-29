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
/// @ingroup geqlf
int64_t unmql(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t k,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* tau,
    std::complex<float>* C, int64_t ldc )
{
    char side_ = to_char( side );
    char trans_ = to_char( trans );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int k_ = to_lapack_int( k );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldc_ = to_lapack_int( ldc );
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_cunmql(
        &side_, &trans_, &m_, &n_, &k_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) C, &ldc_,
        (lapack_complex_float*) qry_work, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );

    LAPACK_cunmql(
        &side_, &trans_, &m_, &n_, &k_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) C, &ldc_,
        (lapack_complex_float*) &work[0], &lwork_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Multiplies the general m-by-n matrix C by Q from `lapack::geqlf` as follows:
///
/// - side = Left,  trans = NoTrans:   $Q C$
/// - side = Right, trans = NoTrans:   $C Q$
/// - side = Left,  trans = ConjTrans: $Q^H C$
/// - side = Right, trans = ConjTrans: $C Q^H$
///
/// where Q is a unitary matrix defined as the product of k
/// elementary reflectors, as returned by `lapack::geqlf`:
/// \[
///     Q = H(k) \dots H(2) H(1).
/// \]
///
/// Q is of order m if side = Left and of order n if side = Right.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this is an alias for `lapack::ormql`.
///
/// @param[in] side
///     - lapack::Side::Left:  apply $Q$ or $Q^H$ from the Left;
///     - lapack::Side::Right: apply $Q$ or $Q^H$ from the Right.
///
/// @param[in] trans
///     - lapack::Op::NoTrans:   No transpose, apply $Q$;
///     - lapack::Op::ConjTrans: Transpose, apply $Q^H$.
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
///     - If side = Left,  m >= k >= 0;
///     - if side = Right, n >= k >= 0.
///
/// @param[in] A
///     - If side = Left,  the m-by-k matrix A, stored in an lda-by-k array;
///     - if side = Right, the n-by-k matrix A, stored in an lda-by-k array.
///     \n
///     The i-th column must contain the vector which defines the
///     elementary reflector H(i), for i = 1, 2, ..., k, as returned by
///     `lapack::geqlf` in the last k columns of its array argument A.
///
/// @param[in] lda
///     The leading dimension of the array A.
///     - If side = Left,  lda >= max(1,m);
///     - if side = Right, lda >= max(1,n).
///
/// @param[in] tau
///     The vector tau of length k.
///     tau(i) must contain the scalar factor of the elementary
///     reflector H(i), as returned by `lapack::geqlf`.
///
/// @param[in,out] C
///     The m-by-n matrix C, stored in an ldc-by-n array.
///     On entry, the m-by-n matrix C.
///     On exit, C is overwritten by $Q C$ or $Q^H C$ or $C Q^H$ or $C Q$.
///
/// @param[in] ldc
///     The leading dimension of the array C. ldc >= max(1,m).
///
/// @return = 0: successful exit
///
/// @ingroup geqlf
int64_t unmql(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t k,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* tau,
    std::complex<double>* C, int64_t ldc )
{
    char side_ = to_char( side );
    char trans_ = to_char( trans );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int k_ = to_lapack_int( k );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldc_ = to_lapack_int( ldc );
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_zunmql(
        &side_, &trans_, &m_, &n_, &k_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) C, &ldc_,
        (lapack_complex_double*) qry_work, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );

    LAPACK_zunmql(
        &side_, &trans_, &m_, &n_, &k_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) C, &ldc_,
        (lapack_complex_double*) &work[0], &lwork_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
