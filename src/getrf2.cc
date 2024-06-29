// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#if LAPACK_VERSION >= 30600  // >= v3.6

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gesv_computational
int64_t getrf2(
    int64_t m, int64_t n,
    float* A, int64_t lda,
    int64_t* ipiv )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( max( 1, min( m, n )) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    LAPACK_sgetrf2(
        &m_, &n_,
        A, &lda_,
        ipiv_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesv_computational
int64_t getrf2(
    int64_t m, int64_t n,
    double* A, int64_t lda,
    int64_t* ipiv )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( max( 1, min( m, n )) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    LAPACK_dgetrf2(
        &m_, &n_,
        A, &lda_,
        ipiv_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesv_computational
int64_t getrf2(
    int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda,
    int64_t* ipiv )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( max( 1, min( m, n )) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    LAPACK_cgetrf2(
        &m_, &n_,
        (lapack_complex_float*) A, &lda_,
        ipiv_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes an LU factorization of a general m-by-n matrix A
/// using partial pivoting with row interchanges.
///
/// The factorization has the form
/// \[
///     A = P L U
/// \]
/// where P is a permutation matrix, L is lower triangular with unit
/// diagonal elements (lower trapezoidal if m > n), and U is upper
/// triangular (upper trapezoidal if m < n).
///
/// This is the recursive version of the algorithm. It divides
/// the matrix into four submatrices:
/// \[
///     A = \begin{bmatrix}
///             A_{11}  &  A_{12}
///         \\  A_{21}  &  A_{22}
///     \end{bmatrix}
/// \]
/// where $A_{11}$ is n1-by-n1 and $A_{22}$ is n2-by-n2,
/// with n1 = min(m,n)/2 and n2 = n-n1.
/// The subroutine calls itself to factor
/// \[
///     \begin{bmatrix}
///             A_{11}
///         \\  A_{21}
///     \end{bmatrix},
/// \]
/// does the swaps on
/// \[
///     \begin{bmatrix}
///             A_{12}
///         \\  A_{22}
///     \end{bmatrix},
/// \]
/// solves $A_{12},$
/// updates $A_{22},$
/// calls itself to factor $A_{22},$
/// and does the swaps on $A_{21}.$
///
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] m
///     The number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix A. n >= 0.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     On entry, the m-by-n matrix to be factored.
///     On exit, the factors L and U from the factorization
///     $A = P L U;$ the unit diagonal elements of L are not stored.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @param[out] ipiv
///     The vector ipiv of length min(m,n).
///     The pivot indices; for 1 <= i <= min(m,n), row i of the
///     matrix was interchanged with row ipiv(i).
///
/// @return = 0: successful exit
/// @return > 0: if return value = i, U(i,i) is exactly zero. The factorization
///     has been completed, but the factor U is exactly
///     singular, and division by zero will occur if it is used
///     to solve a system of equations.
///
/// @ingroup gesv_computational
int64_t getrf2(
    int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda,
    int64_t* ipiv )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( (min(m,n)) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    LAPACK_zgetrf2(
        &m_, &n_,
        (lapack_complex_double*) A, &lda_,
        ipiv_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= v3.6
