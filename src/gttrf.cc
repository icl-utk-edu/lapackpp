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
/// @ingroup gtsv_computational
int64_t gttrf(
    int64_t n,
    float* DL,
    float* D,
    float* DU,
    float* DU2,
    int64_t* ipiv )
{
    lapack_int n_ = to_lapack_int( n );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( (n) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    LAPACK_sgttrf(
        &n_,
        DL,
        D,
        DU,
        DU2,
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
/// @ingroup gtsv_computational
int64_t gttrf(
    int64_t n,
    double* DL,
    double* D,
    double* DU,
    double* DU2,
    int64_t* ipiv )
{
    lapack_int n_ = to_lapack_int( n );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( (n) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    LAPACK_dgttrf(
        &n_,
        DL,
        D,
        DU,
        DU2,
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
/// @ingroup gtsv_computational
int64_t gttrf(
    int64_t n,
    std::complex<float>* DL,
    std::complex<float>* D,
    std::complex<float>* DU,
    std::complex<float>* DU2,
    int64_t* ipiv )
{
    lapack_int n_ = to_lapack_int( n );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( (n) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    LAPACK_cgttrf(
        &n_,
        (lapack_complex_float*) DL,
        (lapack_complex_float*) D,
        (lapack_complex_float*) DU,
        (lapack_complex_float*) DU2,
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
/// Computes an LU factorization of a complex tridiagonal matrix A
/// using elimination with partial pivoting and row interchanges.
///
/// The factorization has the form
/// \[
///     A = L U
/// \]
/// where L is a product of permutation and unit lower bidiagonal
/// matrices and U is upper triangular with nonzeros in only the main
/// diagonal and first two superdiagonals.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] n
///     The order of the matrix A.
///
/// @param[in,out] DL
///     The vector DL of length n-1.
///     On entry, DL must contain the (n-1) sub-diagonal elements of A.
///     \n
///     On exit, DL is overwritten by the (n-1) multipliers that
///     define the matrix L from the LU factorization of A.
///
/// @param[in,out] D
///     The vector D of length n.
///     On entry, D must contain the diagonal elements of A.
///     \n
///     On exit, D is overwritten by the n diagonal elements of the
///     upper triangular matrix U from the LU factorization of A.
///
/// @param[in,out] DU
///     The vector DU of length n-1.
///     On entry, DU must contain the (n-1) super-diagonal elements of A.
///     \n
///     On exit, DU is overwritten by the (n-1) elements of the first
///     super-diagonal of U.
///
/// @param[out] DU2
///     The vector DU2 of length n-2.
///     On exit, DU2 is overwritten by the (n-2) elements of the
///     second super-diagonal of U.
///
/// @param[out] ipiv
///     The vector ipiv of length n.
///     The pivot indices; for 1 <= i <= n, row i of the matrix was
///     interchanged with row ipiv(i). ipiv(i) will always be either
///     i or i+1; ipiv(i) = i indicates a row interchange was not
///     required.
///
/// @return = 0: successful exit
/// @return > 0: if return value = i, U(i,i) is exactly zero. The factorization
///     has been completed, but the factor U is exactly
///     singular, and division by zero will occur if it is used
///     to solve a system of equations.
///
/// @ingroup gtsv_computational
int64_t gttrf(
    int64_t n,
    std::complex<double>* DL,
    std::complex<double>* D,
    std::complex<double>* DU,
    std::complex<double>* DU2,
    int64_t* ipiv )
{
    lapack_int n_ = to_lapack_int( n );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( (n) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    LAPACK_zgttrf(
        &n_,
        (lapack_complex_double*) DL,
        (lapack_complex_double*) D,
        (lapack_complex_double*) DU,
        (lapack_complex_double*) DU2,
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
