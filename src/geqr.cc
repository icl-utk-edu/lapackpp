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
/// @ingroup geqrf
int64_t geqr(
    int64_t m, int64_t n,
    float* A, int64_t lda,
    float* T, int64_t tsize )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(tsize) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int tsize_ = (lapack_int) tsize;
    lapack_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_sgeqr(
        &m_, &n_,
        A, &lda_,
        T, &tsize_,
        qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    // tsize == -1 or -2 is query
    if (tsize < 0) {
        return info_;
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< float > work( lwork_ );

    LAPACK_sgeqr(
        &m_, &n_,
        A, &lda_,
        T, &tsize_,
        &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geqrf
int64_t geqr(
    int64_t m, int64_t n,
    double* A, int64_t lda,
    double* T, int64_t tsize )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(tsize) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int tsize_ = (lapack_int) tsize;
    lapack_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_dgeqr(
        &m_, &n_,
        A, &lda_,
        T, &tsize_,
        qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    // tsize == -1 or -2 is query
    if (tsize < 0) {
        return info_;
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< double > work( lwork_ );

    LAPACK_dgeqr(
        &m_, &n_,
        A, &lda_,
        T, &tsize_,
        &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geqrf
int64_t geqr(
    int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* T, int64_t tsize )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(tsize) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int tsize_ = (lapack_int) tsize;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_cgeqr(
        &m_, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) T, &tsize_,
        (lapack_complex_float*) qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    // tsize == -1 or -2 is query
    if (tsize < 0) {
        return info_;
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );

    LAPACK_cgeqr(
        &m_, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) T, &tsize_,
        (lapack_complex_float*) &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes a QR factorization of an m-by-n matrix A:
/// \[
///     A = Q \begin{bmatrix} R
///           \\              0
///           \end{bmatrix},
/// \]
/// Q is a m-by-m orthogonal matrix;
/// R is an upper-triangular n-by-n matrix;
/// 0 is a (m - n)-by-n zero matrix, if m > n.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @since LAPACK 3.7.0
///
/// @param[in] m
///     The number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix A. n >= 0.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     On entry, the m-by-n matrix A.
///     On exit, the elements on and above the diagonal of the array
///     contain the min(m,n)-by-n upper trapezoidal matrix R
///     (R is upper triangular if m >= n);
///     the elements below the diagonal are used to store part of the
///     data structure to represent Q.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @param[out] T
///     The vector T of length max(5,tsize).
///     On successful exit, T[0] returns optimal (or either minimal
///     or optimal, if query is assumed) tsize. See tsize for details.
///     Remaining T contains part of the data structure used to represent Q.
///     If one wants to apply or construct Q, then one needs to keep T
///     (in addition to A) and pass it to further subroutines.
///
/// @param[in] tsize
///     If tsize >= 5, the dimension of the array T.
///     If tsize = -1 or -2, then a workspace query is assumed. The routine
///     only calculates the sizes of the T array, returns this
///     value as the first entries of the T array, and no error
///     message related to T is issued.
///     If tsize = -1, the routine calculates optimal size of T for the
///     optimum performance and returns this value in T[0].
///     If tsize = -2, the routine calculates minimal size of T and
///     returns this value in T[0].
///
/// @retval = 0: successful exit
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The goal of the interface is to give maximum freedom to the developers for
/// creating any QR factorization algorithm they wish. The
/// trapezoidal R has to be stored in the upper part of A. The lower part of A
/// and the array T can be used to store any relevant information for applying or
/// constructing the Q factor.
///
/// Caution: One should not expect the size of T to be the same from one
/// LAPACK implementation to the other, or even from one execution to the other.
/// A workspace query for T is needed at each execution. However,
/// for a given execution, the size of T are fixed and will not change
/// from one query to the next.
///
// -----------------------------------------------------------------------------
/// @par Further Details particular to the Netlib LAPACK implementation
///
/// These details are particular for the Netlib LAPACK implementation.
/// Users should not take them for granted. These details may change in
/// the future, and are not likely true for another LAPACK
/// implementation. These details are relevant if one wants to try to
/// understand the code. They are not part of the interface.
///
/// In this version,
///
///     T[1]: row block size (mb)
///     T[2]: column block size (nb)
///     T[5:TSIZE-1]: data structure needed for Q, computed by latsqr or geqrt
///
/// Depending on the matrix dimensions m and n, and row and column
/// block sizes mb and nb returned by ilaenv, geqr will use either
/// latsqr (if the matrix is tall-and-skinny) or geqrt to compute
/// the QR factorization.
///
/// @ingroup geqrf
int64_t geqr(
    int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* T, int64_t tsize )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(tsize) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int tsize_ = (lapack_int) tsize;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_zgeqr(
        &m_, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) T, &tsize_,
        (lapack_complex_double*) qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    // tsize == -1 or -2 is query
    if (tsize < 0) {
        return info_;
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );

    LAPACK_zgeqr(
        &m_, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) T, &tsize_,
        (lapack_complex_double*) &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.7.0
