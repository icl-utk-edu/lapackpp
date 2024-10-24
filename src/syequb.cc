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
/// @ingroup sysv_computational
int64_t syequb(
    lapack::Uplo uplo, int64_t n,
    float const* A, int64_t lda,
    float* S,
    float* scond,
    float* amax )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (2*n) );

    LAPACK_ssyequb(
        &uplo_, &n_,
        A, &lda_,
        S, scond, amax,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup sysv_computational
int64_t syequb(
    lapack::Uplo uplo, int64_t n,
    double const* A, int64_t lda,
    double* S,
    double* scond,
    double* amax )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (2*n) );

    LAPACK_dsyequb(
        &uplo_, &n_,
        A, &lda_,
        S, scond, amax,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup sysv_computational
int64_t syequb(
    lapack::Uplo uplo, int64_t n,
    std::complex<float> const* A, int64_t lda,
    float* S,
    float* scond,
    float* amax )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (2*n) );

    LAPACK_csyequb(
        &uplo_, &n_,
        (lapack_complex_float*) A, &lda_,
        S, scond, amax,
        (lapack_complex_float*) &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes row and column scalings intended to equilibrate a
/// symmetric matrix A (with respect to the Euclidean norm) and reduce
/// its condition number. The scale factors S are computed by the BIN
/// algorithm (see references) so that the scaled matrix B with elements
/// $B_{i,j} = S_{i} A_{i,j} S_{j}$ has a condition number within a factor n of
/// the smallest possible condition number over all possible diagonal
/// scalings.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, `lapack::heequb` is an alias for this.
/// For complex Hermitian matrices, see `lapack::heequb`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangle of A is stored;
///     - lapack::Uplo::Lower: Lower triangle of A is stored.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     The n-by-n symmetric matrix whose scaling factors are to be
///     computed.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[out] S
///     The vector S of length n.
///     If successful, S contains the scale factors for A.
///
/// @param[out] scond
///     If successful, S contains the ratio of the smallest S(i) to
///     the largest S(i). If scond >= 0.1 and amax is neither too
///     large nor too small, it is not worth scaling by S.
///
/// @param[out] amax
///     Largest absolute value of any matrix element. If amax is
///     very close to overflow or very close to underflow, the
///     matrix should be scaled.
///
/// @return = 0: successful exit
/// @return > 0: if return value = i, the i-th diagonal element is nonpositive.
///
/// @ingroup sysv_computational
int64_t syequb(
    lapack::Uplo uplo, int64_t n,
    std::complex<double> const* A, int64_t lda,
    double* S,
    double* scond,
    double* amax )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (2*n) );

    LAPACK_zsyequb(
        &uplo_, &n_,
        (lapack_complex_double*) A, &lda_,
        S, scond, amax,
        (lapack_complex_double*) &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
