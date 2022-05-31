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
/// @ingroup gesvd_computational
int64_t gebrd(
    int64_t m, int64_t n,
    float* A, int64_t lda,
    float* D,
    float* E,
    float* tauq,
    float* taup )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_sgebrd(
        &m_, &n_,
        A, &lda_,
        D,
        E,
        tauq,
        taup,
        qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< float > work( lwork_ );

    LAPACK_sgebrd(
        &m_, &n_,
        A, &lda_,
        D,
        E,
        tauq,
        taup,
        &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesvd_computational
int64_t gebrd(
    int64_t m, int64_t n,
    double* A, int64_t lda,
    double* D,
    double* E,
    double* tauq,
    double* taup )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_dgebrd(
        &m_, &n_,
        A, &lda_,
        D,
        E,
        tauq,
        taup,
        qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< double > work( lwork_ );

    LAPACK_dgebrd(
        &m_, &n_,
        A, &lda_,
        D,
        E,
        tauq,
        taup,
        &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesvd_computational
int64_t gebrd(
    int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda,
    float* D,
    float* E,
    std::complex<float>* tauq,
    std::complex<float>* taup )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_cgebrd(
        &m_, &n_,
        (lapack_complex_float*) A, &lda_,
        D,
        E,
        (lapack_complex_float*) tauq,
        (lapack_complex_float*) taup,
        (lapack_complex_float*) qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );

    LAPACK_cgebrd(
        &m_, &n_,
        (lapack_complex_float*) A, &lda_,
        D,
        E,
        (lapack_complex_float*) tauq,
        (lapack_complex_float*) taup,
        (lapack_complex_float*) &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Reduces a general m-by-n matrix A to upper or lower
/// bidiagonal form B by a unitary transformation: $Q^H A P = B$.
///
/// If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] m
///     The number of rows in the matrix A. m >= 0.
///
/// @param[in] n
///     The number of columns in the matrix A. n >= 0.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     On entry, the m-by-n general matrix to be reduced.
///     On exit:
///     - If m >= n, the diagonal and the first superdiagonal are
///     overwritten with the upper bidiagonal matrix B; the
///     elements below the diagonal, with the array tauq, represent
///     the unitary matrix Q as a product of elementary
///     reflectors, and the elements above the first superdiagonal,
///     with the array taup, represent the unitary matrix P as
///     a product of elementary reflectors;
///
///     - If m < n, the diagonal and the first subdiagonal are
///     overwritten with the lower bidiagonal matrix B; the
///     elements below the first subdiagonal, with the array tauq,
///     represent the unitary matrix Q as a product of
///     elementary reflectors, and the elements above the diagonal,
///     with the array taup, represent the unitary matrix P as
///     a product of elementary reflectors.
///     See Further Details.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @param[out] D
///     The vector D of length min(m,n).
///     The diagonal elements of the bidiagonal matrix B:
///     D(i) = A(i,i).
///
/// @param[out] E
///     The vector E of length min(m,n)-1.
///     The off-diagonal elements of the bidiagonal matrix B:
///     if m >= n, E(i) = A(i,i+1) for i = 1, 2, ..., n-1;
///     if m <  n, E(i) = A(i+1,i) for i = 1, 2, ..., m-1.
///
/// @param[out] tauq
///     The vector tauq of length min(m,n).
///     The scalar factors of the elementary reflectors which
///     represent the unitary matrix Q. See Further Details.
///
/// @param[out] taup
///     The vector taup of length min(m,n).
///     The scalar factors of the elementary reflectors which
///     represent the unitary matrix P. See Further Details.
///
/// @return = 0: successful exit.
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The matrices Q and P are represented as products of elementary
/// reflectors:
///
/// If m >= n,
/// \[
///     Q = H(1) H(2) . . . H(n)
/// \]
//  and
/// \[
///     P = G(1) G(2) . . . G(n-1).
/// \]
///
/// Each H(i) and G(i) has the form:
/// \[
///     H(i) = I - \tau_q v v^H
/// \]
/// and
/// \[
///     G(i) = I - \tau_p u u^H
/// \]
///
/// where $\tau_q$ and $\tau_p$ are scalars, and v and u are
/// vectors; v(1:i-1) = 0, v(i) = 1, and v(i+1:m) is stored on exit in
/// A(i+1:m,i); u(1:i) = 0, u(i+1) = 1, and u(i+2:n) is stored on exit in
/// A(i,i+2:n); $\tau_q$ is stored in tauq(i) and $\tau_p$ in taup(i).
///
/// If m < n,
/// \[
///     Q = H(1) H(2) . . . H(m-1)
/// \]
/// and
/// \[
///     P = G(1) G(2) . . . G(m)
/// \]
///
/// Each H(i) and G(i) has the form:
/// \[
///     H(i) = I - \tau_q v v^H
/// \]
/// and
/// \[
///     G(i) = I - \tau_p u u^H
/// \]
///
/// where $\tau_q$ and $\tau_p$ are scalars, and v and u are
/// vectors; v(1:i) = 0, v(i+1) = 1, and v(i+2:m) is stored on exit in
/// A(i+2:m,i); u(1:i-1) = 0, u(i) = 1, and u(i+1:n) is stored on exit in
/// A(i,i+1:n); $\tau_q$ is stored in tauq(i) and $\tau_p$ in taup(i).
///
/// The contents of A on exit are illustrated by the following examples:
///
///     m = 6 and n = 5 (m >= n):          m = 5 and n = 6 (m < n):
///
///     (  d   e   u1  u1  u1 )            (  d   u1  u1  u1  u1  u1 )
///     (  v1  d   e   u2  u2 )            (  e   d   u2  u2  u2  u2 )
///     (  v1  v2  d   e   u3 )            (  v1  e   d   u3  u3  u3 )
///     (  v1  v2  v3  d   e  )            (  v1  v2  e   d   u4  u4 )
///     (  v1  v2  v3  v4  d  )            (  v1  v2  v3  e   d   u5 )
///     (  v1  v2  v3  v4  v5 )
///
/// where d and e denote diagonal and off-diagonal elements of B, vi
/// denotes an element of the vector defining H(i), and ui an element of
/// the vector defining G(i).
///
/// @ingroup gesvd_computational
int64_t gebrd(
    int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda,
    double* D,
    double* E,
    std::complex<double>* tauq,
    std::complex<double>* taup )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_zgebrd(
        &m_, &n_,
        (lapack_complex_double*) A, &lda_,
        D,
        E,
        (lapack_complex_double*) tauq,
        (lapack_complex_double*) taup,
        (lapack_complex_double*) qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );

    LAPACK_zgebrd(
        &m_, &n_,
        (lapack_complex_double*) A, &lda_,
        D,
        E,
        (lapack_complex_double*) tauq,
        (lapack_complex_double*) taup,
        (lapack_complex_double*) &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
