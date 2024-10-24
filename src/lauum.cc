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
int64_t lauum(
    lapack::Uplo uplo, int64_t n,
    float* A, int64_t lda )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_slauum(
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
int64_t lauum(
    lapack::Uplo uplo, int64_t n,
    double* A, int64_t lda )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_dlauum(
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
int64_t lauum(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_clauum(
        &uplo_, &n_,
        (lapack_complex_float*) A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the product $U U^H$ or $L^H L,$ where the triangular
/// factor U or L is stored in the upper or lower triangular part of
/// the array A.
///
/// If uplo = Upper then the upper triangle of the result is stored,
/// overwriting the factor U in A.
///
/// If uplo = Lower then the lower triangle of the result is stored,
/// overwriting the factor L in A.
///
/// This is the blocked form of the algorithm, calling Level 3 BLAS.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] uplo
///     Whether the triangular factor stored in the array A
///     is upper or lower triangular:
///     - lapack::Uplo::Upper: Upper triangular
///     - lapack::Uplo::Lower: Lower triangular
///
/// @param[in] n
///     The order of the triangular factor U or L. n >= 0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     On entry, the triangular factor U or L.
///     On exit,
///     - if uplo = Upper, the upper triangle of A is
///     overwritten with the upper triangle of the product $U U^H$;
///     - if uplo = Lower, the lower triangle of A is overwritten with
///     the lower triangle of the product $L^H L$.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @return = 0: successful exit
///
/// @ingroup posv_computational
int64_t lauum(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_zlauum(
        &uplo_, &n_,
        (lapack_complex_double*) A, &lda_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
