// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"

#if LAPACK_VERSION >= 30301  // >= 3.3.1

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup hesv_computational
void heswapr(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda, int64_t i1, int64_t i2 )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int i1_ = to_lapack_int( i1 );
    lapack_int i2_ = to_lapack_int( i2 );

    LAPACK_cheswapr(
        &uplo_, &n_,
        (lapack_complex_float*) A, &lda_, &i1_, &i2_
    );
}

// -----------------------------------------------------------------------------
/// Applies an elementary permutation on the rows and the columns of
/// a Hermitian matrix.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this in an alias for `lapack::syswapr`.
/// For complex symmetric matrices, see `lapack::syswapr`.
///
/// @param[in] uplo
///     Whether the details of the factorization are stored
///     as an upper or lower triangular matrix.
///     - lapack::Uplo::Upper: Upper triangular, form is $A = U D U^T;$
///     - lapack::Uplo::Lower: Lower triangular, form is $A = L D L^T.$
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     TODO (the LAPACK documentation seems wrong).
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[in] i1
///     Index of the first row to swap
///
/// @param[in] i2
///     Index of the second row to swap
///
/// @ingroup hesv_computational
void heswapr(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda, int64_t i1, int64_t i2 )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int i1_ = to_lapack_int( i1 );
    lapack_int i2_ = to_lapack_int( i2 );

    LAPACK_zheswapr(
        &uplo_, &n_,
        (lapack_complex_double*) A, &lda_, &i1_, &i2_
    );
}

}  // namespace lapack

#endif  // LAPACK >= 3.3.1
