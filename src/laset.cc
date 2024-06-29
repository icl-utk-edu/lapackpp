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
/// @ingroup initialize
void laset(
    lapack::MatrixType matrixtype, int64_t m, int64_t n, float offdiag, float diag,
    float* A, int64_t lda )
{
    char matrixtype_ = to_char( matrixtype );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );

    LAPACK_slaset(
        &matrixtype_, &m_, &n_, &offdiag, &diag,
        A, &lda_
    );
}

// -----------------------------------------------------------------------------
/// @ingroup initialize
void laset(
    lapack::MatrixType matrixtype, int64_t m, int64_t n, double offdiag, double diag,
    double* A, int64_t lda )
{
    char matrixtype_ = to_char( matrixtype );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );

    LAPACK_dlaset(
        &matrixtype_, &m_, &n_, &offdiag, &diag,
        A, &lda_
    );
}

// -----------------------------------------------------------------------------
/// @ingroup initialize
void laset(
    lapack::MatrixType matrixtype, int64_t m, int64_t n, std::complex<float> offdiag, std::complex<float> diag,
    std::complex<float>* A, int64_t lda )
{
    char matrixtype_ = to_char( matrixtype );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );

    LAPACK_claset(
        &matrixtype_, &m_, &n_, (lapack_complex_float*) &offdiag, (lapack_complex_float*) &diag,
        (lapack_complex_float*) A, &lda_
    );
}

// -----------------------------------------------------------------------------
/// Initializes a 2-D array A to diag on the diagonal and
/// offdiag on the offdiagonals.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] matrixtype
///     Specifies the part of the matrix A to be set.
///     - lapack::MatrixType::Upper:
///         Upper triangular part is set. The lower triangle is unchanged.
///
///     - lapack::MatrixType::Lower:
///         Lower triangular part is set. The upper triangle is unchanged.
///
///     - lapack::MatrixType::General:
///         All of the matrix A is set.
///
/// @param[in] m
///     On entry, m specifies the number of rows of A.
///
/// @param[in] n
///     On entry, n specifies the number of columns of A.
///
/// @param[in] offdiag
///     All the offdiagonal array elements are set to offdiag.
///
/// @param[in] diag
///     All the diagonal array elements are set to diag.
///
/// @param[out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     On entry, the m-by-n matrix A.
///     On exit,
///     - A(i,j) = offdiag, 1 <= i <= m, 1 <= j <= n, i != j;
///     - A(i,i) = diag, 1 <= i <= min(m,n)
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @ingroup initialize
void laset(
    lapack::MatrixType matrixtype, int64_t m, int64_t n, std::complex<double> offdiag, std::complex<double> diag,
    std::complex<double>* A, int64_t lda )
{
    char matrixtype_ = to_char( matrixtype );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );

    LAPACK_zlaset(
        &matrixtype_, &m_, &n_, (lapack_complex_double*) &offdiag, (lapack_complex_double*) &diag,
        (lapack_complex_double*) A, &lda_
    );
}

}  // namespace lapack
