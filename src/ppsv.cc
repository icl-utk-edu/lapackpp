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
/// @ingroup ppsv
int64_t ppsv(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    float* AP,
    float* B, int64_t ldb )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_sppsv(
        &uplo_, &n_, &nrhs_,
        AP,
        B, &ldb_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ppsv
int64_t ppsv(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    double* AP,
    double* B, int64_t ldb )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_dppsv(
        &uplo_, &n_, &nrhs_,
        AP,
        B, &ldb_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ppsv
int64_t ppsv(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<float>* AP,
    std::complex<float>* B, int64_t ldb )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_cppsv(
        &uplo_, &n_, &nrhs_,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) B, &ldb_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the solution to a system of linear equations
/// \[
///     A X = B,
/// \]
/// where A is an n-by-n Hermitian positive definite matrix stored in
/// packed format and X and B are n-by-nrhs matrices.
///
/// The Cholesky decomposition is used to factor A as
///     $A = U^H U$ if uplo = Upper, or
///     $A = L L^H$ if uplo = Lower,
/// where U is an upper triangular matrix and L is a lower triangular
/// matrix. The factored form of A is then used to solve the system of
/// equations $A X = B.$
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangle of A is stored;
///     - lapack::Uplo::Lower: Lower triangle of A is stored.
///
/// @param[in] n
///     The number of linear equations, i.e., the order of the
///     matrix A. n >= 0.
///
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrix B. nrhs >= 0.
///
/// @param[in,out] AP
///     The vector AP of length n*(n+1)/2.
///     - On entry, the upper or lower triangle of the Hermitian matrix
///     A, packed columnwise in a linear array. The j-th column of A
///     is stored in the array AP as follows:
///       - if uplo = Upper, AP(i + (j-1)*j/2) = A(i,j) for 1 <= i <= j;
///       - if uplo = Lower, AP(i + (j-1)*(2n-j)/2) = A(i,j) for j <= i <= n.
///         \n
///         See below for further details.
///
///     - On successful exit, the factor U or L from the Cholesky
///     factorization $A = U^H U$ or $A = L L^H,$ in the same storage
///     format as A.
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
/// @return > 0: if return value = i, the leading minor of order i of A is not
///     positive definite, so the factorization could not be
///     completed, and the solution has not been computed.
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The packed storage scheme is illustrated by the following example
/// when n = 4, uplo = Upper:
///
/// Two-dimensional storage of the Hermitian matrix A:
///
///     [ a11 a12 a13 a14 ]
///     [     a22 a23 a24 ]
///     [         a33 a34 ]    (aij = conj(aji))
///     [             a44 ]
///
/// Packed storage of the upper triangle of A:
///
///     AP = [ a11, a12, a22, a13, a23, a33, a14, a24, a34, a44 ]
///
/// @ingroup ppsv
int64_t ppsv(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<double>* AP,
    std::complex<double>* B, int64_t ldb )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_zppsv(
        &uplo_, &n_, &nrhs_,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) B, &ldb_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
