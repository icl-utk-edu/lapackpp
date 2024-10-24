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
/// @ingroup norm
float lanhs(
    lapack::Norm norm, int64_t n,
    float const* A, int64_t lda )
{
    char norm_ = to_char( norm );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );

    // from docs
    int64_t lwork = (norm == Norm::Inf ? n : 1);

    // allocate workspace
    lapack::vector< float > work( max(1,lwork) );

    return LAPACK_slanhs(
        &norm_, &n_,
        A, &lda_,
        &work[0]
    );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
double lanhs(
    lapack::Norm norm, int64_t n,
    double const* A, int64_t lda )
{
    char norm_ = to_char( norm );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );

    // from docs
    int64_t lwork = (norm == Norm::Inf ? n : 1);

    // allocate workspace
    lapack::vector< double > work( max(1,lwork) );

    return LAPACK_dlanhs(
        &norm_, &n_,
        A, &lda_,
        &work[0]
    );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
float lanhs(
    lapack::Norm norm, int64_t n,
    std::complex<float> const* A, int64_t lda )
{
    char norm_ = to_char( norm );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );

    // from docs
    int64_t lwork = (norm == Norm::Inf ? n : 1);

    // allocate workspace
    lapack::vector< float > work( max(1,lwork) );

    return LAPACK_clanhs(
        &norm_, &n_,
        (lapack_complex_float*) A, &lda_,
        &work[0]
    );
}

// -----------------------------------------------------------------------------
/// Returns the value of the one norm, Frobenius norm,
/// infinity norm, or the element of largest absolute value of a
/// Hessenberg matrix A.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] norm
///     The value to be returned:
///     - lapack::Norm::Max: max norm: max(abs(A(i,j))).
///                          Note this is not a consistent matrix norm.
///     - lapack::Norm::One: one norm: maximum column sum
///     - lapack::Norm::Inf: infinity norm: maximum row sum
///     - lapack::Norm::Fro: Frobenius norm: square root of sum of squares
///
/// @param[in] n
///     The order of the matrix A. n >= 0. When n = 0, returns zero.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     The n-by-n upper Hessenberg matrix A; the part of A below the
///     first sub-diagonal is not referenced.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(n,1).
///
/// @ingroup norm
double lanhs(
    lapack::Norm norm, int64_t n,
    std::complex<double> const* A, int64_t lda )
{
    char norm_ = to_char( norm );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );

    // from docs
    int64_t lwork = (norm == Norm::Inf ? n : 1);

    // allocate workspace
    lapack::vector< double > work( max(1,lwork) );

    return LAPACK_zlanhs(
        &norm_, &n_,
        (lapack_complex_double*) A, &lda_,
        &work[0]
    );
}

}  // namespace lapack
