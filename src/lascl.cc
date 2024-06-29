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
/// @ingroup auxiliary
int64_t lascl(
    lapack::MatrixType matrixtype, int64_t kl, int64_t ku, float cfrom, float cto, int64_t m, int64_t n,
    float* A, int64_t lda )
{
    char matrixtype_ = to_char( matrixtype );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_slascl(
        &matrixtype_, &kl_, &ku_, &cfrom, &cto, &m_, &n_,
        A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup auxiliary
int64_t lascl(
    lapack::MatrixType matrixtype, int64_t kl, int64_t ku, double cfrom, double cto, int64_t m, int64_t n,
    double* A, int64_t lda )
{
    char matrixtype_ = to_char( matrixtype );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_dlascl(
        &matrixtype_, &kl_, &ku_, &cfrom, &cto, &m_, &n_,
        A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup auxiliary
int64_t lascl(
    lapack::MatrixType matrixtype, int64_t kl, int64_t ku, float cfrom, float cto, int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda )
{
    char matrixtype_ = to_char( matrixtype );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_clascl(
        &matrixtype_, &kl_, &ku_, &cfrom, &cto, &m_, &n_,
        (lapack_complex_float*) A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Multiplies the m-by-n complex matrix A by the real scalar
/// cto / cfrom. This is done without over/underflow as long as the final
/// result cto * A(i,j) / cfrom does not over/underflow. type specifies that
/// A may be full, upper triangular, lower triangular, upper Hessenberg,
/// or banded.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] matrixtype
///     type indices the storage type of the input matrix.
///     - lapack::MatrixType::General:
///         A is a full matrix.
///
///     - lapack::MatrixType::Lower:
///         A is a lower triangular matrix.
///
///     - lapack::MatrixType::Upper:
///         A is an upper triangular matrix.
///
///     - lapack::MatrixType::Hessenberg:
///         A is an upper Hessenberg matrix.
///
///     - lapack::MatrixType::LowerBand:
///         A is a symmetric band matrix with lower bandwidth kl
///         and upper bandwidth ku and with the only the lower
///         half stored.
///
///     - lapack::MatrixType::UpperBand:
///         A is a symmetric band matrix with lower bandwidth kl
///         and upper bandwidth ku and with the only the upper
///         half stored.
///
///     - lapack::MatrixType::Band:
///         A is a band matrix with lower bandwidth kl and upper
///         bandwidth ku. See `lapack::gbtrf` for storage details.
///
/// @param[in] kl
///     The lower bandwidth of A.
///     Referenced only if type = LowerBand, UpperBand, or Band.
///
/// @param[in] ku
///     The upper bandwidth of A.
///     Referenced only if type = LowerBand, UpperBand, or Band.
///
/// @param[in] cfrom
///
/// @param[in] cto
///     The matrix A is multiplied by cto/cfrom. A(i,j) is computed
///     without over/underflow if the final result cto*A(i,j)/cfrom
///     can be represented without over/underflow. cfrom must be
///     nonzero.
///
/// @param[in] m
///     The number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix A. n >= 0.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     The matrix to be multiplied by cto/cfrom. See matrixtype for the
///     storage type.
///
/// @param[in] lda
///     The leading dimension of the array A.
///     - If matrixtype = General, Lower, Upper, or Hessenberg, lda >= max(1,m);
///     - if matrixtype = LowerBand, lda >= kl+1;
///     - if matrixtype = UpperBand, lda >= ku+1;
///     - if matrixtype = Band, lda >= 2*kl+ku+1.
///
/// @return = 0: successful exit
///
/// @ingroup auxiliary
int64_t lascl(
    lapack::MatrixType matrixtype, int64_t kl, int64_t ku, double cfrom, double cto, int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda )
{
    char matrixtype_ = to_char( matrixtype );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_zlascl(
        &matrixtype_, &kl_, &ku_, &cfrom, &cto, &m_, &n_,
        (lapack_complex_double*) A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
