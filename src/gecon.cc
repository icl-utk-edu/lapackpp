// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gesv_computational
int64_t gecon(
    lapack::Norm norm, int64_t n,
    float const* A, int64_t lda, float anorm,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (4*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_sgecon(
        &norm_, &n_,
        A, &lda_, &anorm, rcond,
        &work[0],
        &iwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesv_computational
int64_t gecon(
    lapack::Norm norm, int64_t n,
    double const* A, int64_t lda, double anorm,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (4*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_dgecon(
        &norm_, &n_,
        A, &lda_, &anorm, rcond,
        &work[0],
        &iwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesv_computational
int64_t gecon(
    lapack::Norm norm, int64_t n,
    std::complex<float> const* A, int64_t lda, float anorm,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (2*n) );
    lapack::vector< float > rwork( (2*n) );

    LAPACK_cgecon(
        &norm_, &n_,
        (lapack_complex_float*) A, &lda_, &anorm, rcond,
        (lapack_complex_float*) &work[0],
        &rwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Estimates the reciprocal of the condition number of a general
/// matrix A, in either the 1-norm or the infinity-norm, using
/// the LU factorization computed by `lapack::getrf`.
///
/// An estimate is obtained for norm(inv(A)), and the reciprocal of the
/// condition number is computed as
///     rcond = 1 / ( norm(A) * norm(inv(A)) ).
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] norm
///     Whether the 1-norm condition number or the
///     infinity-norm condition number is required:
///     - lapack::Norm::One: 1-norm;
///     - lapack::Norm::Inf: Infinity-norm.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     The factors L and U from the factorization $A = P L U$
///     as computed by `lapack::getrf`.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[in] anorm
///     - If norm = One, the 1-norm of the original matrix A.
///     - If norm = Inf, the infinity-norm of the original matrix A.
///
/// @param[out] rcond
///     The reciprocal of the condition number of the matrix A,
///     computed as rcond = 1/(norm(A) * norm(inv(A))).
///
/// @return = 0: successful exit
///
/// @ingroup gesv_computational
int64_t gecon(
    lapack::Norm norm, int64_t n,
    std::complex<double> const* A, int64_t lda, double anorm,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (2*n) );
    lapack::vector< double > rwork( (2*n) );

    LAPACK_zgecon(
        &norm_, &n_,
        (lapack_complex_double*) A, &lda_, &anorm, rcond,
        (lapack_complex_double*) &work[0],
        &rwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
