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
/// @ingroup posv_computational
int64_t potri(
    lapack::Uplo uplo, int64_t n,
    float* A, int64_t lda )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_spotri(
        &uplo_, &n_,
        A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup posv_computational
int64_t potri(
    lapack::Uplo uplo, int64_t n,
    double* A, int64_t lda )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_dpotri(
        &uplo_, &n_,
        A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup posv_computational
int64_t potri(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_cpotri(
        &uplo_, &n_,
        (lapack_complex_float*) A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the inverse of a Hermitian positive definite
/// matrix A using the Cholesky factorization $A = U^H U$ or $A = L L^H$
/// computed by `lapack::potrf`.
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
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     On entry, the triangular factor U or L from the Cholesky
///     factorization $A = U^H U$ or $A = L L^H,$ as computed by
///     `lapack::potrf`.
///     On exit, the upper or lower triangle of the (Hermitian)
///     inverse of A, overwriting the input factor U or L.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @return = 0: successful exit
/// @return > 0: if return value = i, the (i,i) element of the factor U or L is
///     zero, and the inverse could not be computed.
///
/// @ingroup posv_computational
int64_t potri(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_zpotri(
        &uplo_, &n_,
        (lapack_complex_double*) A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
