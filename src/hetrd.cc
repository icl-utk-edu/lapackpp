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
/// @ingroup heev_computational
int64_t hetrd(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda,
    float* D,
    float* E,
    std::complex<float>* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_chetrd(
        &uplo_, &n_,
        (lapack_complex_float*) A, &lda_,
        D,
        E,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) qry_work, &ineg_one, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );

    LAPACK_chetrd(
        &uplo_, &n_,
        (lapack_complex_float*) A, &lda_,
        D,
        E,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Reduces a Hermitian matrix A to real symmetric
/// tridiagonal form T by a unitary similarity transformation:
/// $Q^H A Q = T$.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this is an alias for `lapack::sytrd`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangle of A is stored;
///     - lapack::Uplo::Lower: Lower triangle of A is stored.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     On entry, the Hermitian matrix A.
///     - If uplo = Upper, the leading
///     n-by-n upper triangular part of A contains the upper
///     triangular part of the matrix A, and the strictly lower
///     triangular part of A is not referenced.
///
///     - If uplo = Lower, the
///     leading n-by-n lower triangular part of A contains the lower
///     triangular part of the matrix A, and the strictly upper
///     triangular part of A is not referenced.
///
///     - On exit, if uplo = Upper, the diagonal and first superdiagonal
///     of A are overwritten by the corresponding elements of the
///     tridiagonal matrix T, and the elements above the first
///     superdiagonal, with the array tau, represent the unitary
///     matrix Q as a product of elementary reflectors;
///
///     - On exit, if uplo = Lower, the diagonal and first subdiagonal
///     of A are overwritten by the corresponding elements of the
///     tridiagonal matrix T, and the elements below the first
///     subdiagonal, with the array tau, represent the unitary
///     matrix Q as a product of elementary reflectors. See Further Details.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[out] D
///     The vector D of length n.
///     The diagonal elements of the tridiagonal matrix T:
///     D(i) = A(i,i).
///
/// @param[out] E
///     The vector E of length n-1.
///     The off-diagonal elements of the tridiagonal matrix T:
///     E(i) = A(i,i+1) if uplo = Upper, E(i) = A(i+1,i) if uplo = Lower.
///
/// @param[out] tau
///     The vector tau of length n-1.
///     The scalar factors of the elementary reflectors (see Further Details).
///
/// @return = 0: successful exit
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// If uplo = Upper, the matrix Q is represented as a product of elementary
/// reflectors
/// \[
///     Q = H(n-1) . . . H(2) H(1).
/// \]
///
/// Each H(i) has the form
/// \[
///     H(i) = I - \tau v v^H
/// \]
///
/// where $\tau$ is a scalar, and v is a vector with
/// v(i+1:n) = 0 and v(i) = 1; v(1:i-1) is stored on exit in
/// A(1:i-1,i+1), and $\tau$ in tau(i).
///
/// If uplo = Lower, the matrix Q is represented as a product of elementary
/// reflectors
/// \[
///     Q = H(1) H(2) . . . H(n-1).
/// \]
///
/// Each H(i) has the form
/// \[
///     H(i) = I - \tau v v^H
/// \]
///
/// where $\tau$ is a scalar, and v is a vector with
/// v(1:i) = 0 and v(i+1) = 1; v(i+2:n) is stored on exit in A(i+2:n,i),
/// and $\tau$ in tau(i).
///
/// The contents of A on exit are illustrated by the following examples
/// with n = 5:
///
///     if uplo = Upper:                       if uplo = Lower:
///
///     (  d   e   v2  v3  v4 )              (  d                  )
///     (      d   e   v3  v4 )              (  e   d              )
///     (          d   e   v4 )              (  v1  e   d          )
///     (              d   e  )              (  v1  v2  e   d      )
///     (                  d  )              (  v1  v2  v3  e   d  )
///
/// where d and e denote diagonal and off-diagonal elements of T, and vi
/// denotes an element of the vector defining H(i).
///
/// @ingroup heev_computational
int64_t hetrd(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda,
    double* D,
    double* E,
    std::complex<double>* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_zhetrd(
        &uplo_, &n_,
        (lapack_complex_double*) A, &lda_,
        D,
        E,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) qry_work, &ineg_one, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );

    LAPACK_zhetrd(
        &uplo_, &n_,
        (lapack_complex_double*) A, &lda_,
        D,
        E,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
