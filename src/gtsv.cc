// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gtsv
int64_t gtsv(
    int64_t n, int64_t nrhs,
    float* DL,
    float* D,
    float* DU,
    float* B, int64_t ldb )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_sgtsv(
        &n_, &nrhs_,
        DL,
        D,
        DU,
        B, &ldb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gtsv
int64_t gtsv(
    int64_t n, int64_t nrhs,
    double* DL,
    double* D,
    double* DU,
    double* B, int64_t ldb )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_dgtsv(
        &n_, &nrhs_,
        DL,
        D,
        DU,
        B, &ldb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gtsv
int64_t gtsv(
    int64_t n, int64_t nrhs,
    std::complex<float>* DL,
    std::complex<float>* D,
    std::complex<float>* DU,
    std::complex<float>* B, int64_t ldb )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_cgtsv(
        &n_, &nrhs_,
        (lapack_complex_float*) DL,
        (lapack_complex_float*) D,
        (lapack_complex_float*) DU,
        (lapack_complex_float*) B, &ldb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Solves the equation
/// \[
///     A X = B,
/// \]
/// where A is an n-by-n tridiagonal matrix, by Gaussian elimination with
/// partial pivoting.
///
/// Note that the equation $A^T X = B$ may be solved by interchanging the
/// order of the arguments DU and DL.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrix B. nrhs >= 0.
///
/// @param[in,out] DL
///     The vector DL of length n-1.
///     On entry, DL must contain the (n-1) subdiagonal elements of A.
///     On exit, DL is overwritten by the (n-2) elements of the
///     second superdiagonal of the upper triangular matrix U from
///     the LU factorization of A, in DL(1), ..., DL(n-2).
///
/// @param[in,out] D
///     The vector D of length n.
///     On entry, D must contain the diagonal elements of A.
///     On exit, D is overwritten by the n diagonal elements of U.
///
/// @param[in,out] DU
///     The vector DU of length n-1.
///     On entry, DU must contain the (n-1) superdiagonal elements of A.
///     On exit, DU is overwritten by the (n-1) elements of the first
///     superdiagonal of U.
///
/// @param[in,out] B
///     The n-by-nrhs matrix B, stored in an ldb-by-nrhs array.
///     On entry, the n-by-nrhs right hand side matrix B.
///     On successful exit, the n-by-nrhs solution matrix X.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,n).
///
/// @return = 0: successful exit
/// @return > 0: if return value = i, U(i,i) is exactly zero, and the solution
///     has not been computed. The factorization has not been
///     completed unless i = n.
///
/// @ingroup gtsv
int64_t gtsv(
    int64_t n, int64_t nrhs,
    std::complex<double>* DL,
    std::complex<double>* D,
    std::complex<double>* DU,
    std::complex<double>* B, int64_t ldb )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_zgtsv(
        &n_, &nrhs_,
        (lapack_complex_double*) DL,
        (lapack_complex_double*) D,
        (lapack_complex_double*) DU,
        (lapack_complex_double*) B, &ldb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
