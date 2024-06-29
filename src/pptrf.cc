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
/// @ingroup ppsv_computational
int64_t pptrf(
    lapack::Uplo uplo, int64_t n,
    float* AP )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_spptrf(
        &uplo_, &n_,
        AP, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ppsv_computational
int64_t pptrf(
    lapack::Uplo uplo, int64_t n,
    double* AP )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_dpptrf(
        &uplo_, &n_,
        AP, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ppsv_computational
int64_t pptrf(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* AP )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_cpptrf(
        &uplo_, &n_,
        (lapack_complex_float*) AP, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the Cholesky factorization of a Hermitian
/// positive definite matrix A stored in packed format.
///
/// The factorization has the form
///     $A = U^H U,$ if uplo = Upper, or
///     $A = L L^H,$ if uplo = Lower,
/// where U is an upper triangular matrix and L is lower triangular.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangle of A is stored;
///     - lapack::Uplo::Lower: Lower triangle of A is stored.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
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
///     - On successful exit, the triangular factor U or L from the
///     Cholesky factorization $A = U^H U$ or $A = L L^H,$ in the same
///     storage format as A.
///
/// @return = 0: successful exit
/// @return > 0: if return value = i, the leading minor of order i is not
///     positive definite, and the factorization could not be
///     completed.
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
/// @ingroup ppsv_computational
int64_t pptrf(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* AP )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_zpptrf(
        &uplo_, &n_,
        (lapack_complex_double*) AP, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
