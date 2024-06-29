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
/// @ingroup hesv_computational
int64_t hecon(
    lapack::Uplo uplo, int64_t n,
    std::complex<float> const* A, int64_t lda,
    int64_t const* ipiv, float anorm,
    float* rcond )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int const* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int const* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (2*n) );

    LAPACK_checon(
        &uplo_, &n_,
        (lapack_complex_float*) A, &lda_,
        ipiv_ptr, &anorm, rcond,
        (lapack_complex_float*) &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Estimates the reciprocal of the condition number (in the
/// 1-norm) of a Hermitian matrix A using the factorization
/// $A = U D U^H$ or $A = L D L^H$ computed by `lapack::hetrf`.
///
/// An estimate is obtained for $|| A^{-1} ||_1,$ and the reciprocal of the
/// condition number is computed as $\text{rcond} = 1 / (||A||_1 \cdot || A^{-1} ||_1).$
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this in an alias for `lapack::sycon`.
/// For complex symmetric matrices, see `lapack::sycon`.
///
/// @param[in] uplo
///     Whether the details of the factorization are stored
///     as an upper or lower triangular matrix.
///     - lapack::Uplo::Upper: Upper triangular, form is $A = U D U^H;$
///     - lapack::Uplo::Lower: Lower triangular, form is $A = L D L^H.$
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     The block diagonal matrix D and the multipliers used to
///     obtain the factor U or L as computed by `lapack::hetrf`.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[in] ipiv
///     The vector ipiv of length n.
///     Details of the interchanges and the block structure of D
///     as determined by `lapack::hetrf`.
///
/// @param[in] anorm
///     The 1-norm of the original matrix A.
///
/// @param[out] rcond
///     The reciprocal of the condition number of the matrix A,
///     computed as rcond = 1/(anorm * ainv_norm), where ainv_norm is an
///     estimate of the 1-norm of $A^{-1}$ computed in this routine.
///
/// @return = 0: successful exit
///
/// @ingroup hesv_computational
int64_t hecon(
    lapack::Uplo uplo, int64_t n,
    std::complex<double> const* A, int64_t lda,
    int64_t const* ipiv, double anorm,
    double* rcond )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int const* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int const* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (2*n) );

    LAPACK_zhecon(
        &uplo_, &n_,
        (lapack_complex_double*) A, &lda_,
        ipiv_ptr, &anorm, rcond,
        (lapack_complex_double*) &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
