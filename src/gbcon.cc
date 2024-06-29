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
/// @ingroup gbsv_computational
int64_t gbcon(
    lapack::Norm norm, int64_t n, int64_t kl, int64_t ku,
    float const* AB, int64_t ldab,
    int64_t const* ipiv, float anorm,
    float* rcond )
{
    char norm_ = to_char( norm );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int ldab_ = to_lapack_int( ldab );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int const* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int const* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (3*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_sgbcon(
        &norm_, &n_, &kl_, &ku_,
        AB, &ldab_,
        ipiv_ptr, &anorm, rcond,
        &work[0],
        &iwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gbsv_computational
int64_t gbcon(
    lapack::Norm norm, int64_t n, int64_t kl, int64_t ku,
    double const* AB, int64_t ldab,
    int64_t const* ipiv, double anorm,
    double* rcond )
{
    char norm_ = to_char( norm );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int ldab_ = to_lapack_int( ldab );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int const* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int const* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (3*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_dgbcon(
        &norm_, &n_, &kl_, &ku_,
        AB, &ldab_,
        ipiv_ptr, &anorm, rcond,
        &work[0],
        &iwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gbsv_computational
int64_t gbcon(
    lapack::Norm norm, int64_t n, int64_t kl, int64_t ku,
    std::complex<float> const* AB, int64_t ldab,
    int64_t const* ipiv, float anorm,
    float* rcond )
{
    char norm_ = to_char( norm );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int ldab_ = to_lapack_int( ldab );
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
    lapack::vector< float > rwork( (n) );

    LAPACK_cgbcon(
        &norm_, &n_, &kl_, &ku_,
        (lapack_complex_float*) AB, &ldab_,
        ipiv_ptr, &anorm, rcond,
        (lapack_complex_float*) &work[0],
        &rwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Estimates the reciprocal of the condition number of a
/// general band matrix A, in either the 1-norm or the infinity-norm,
/// using the LU factorization computed by `lapack::gbtrf`.
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
/// @param[in] kl
///     The number of subdiagonals within the band of A. kl >= 0.
///
/// @param[in] ku
///     The number of superdiagonals within the band of A. ku >= 0.
///
/// @param[in] AB
///     The n-by-n band matrix AB, stored in an ldab-by-n array.
///     Details of the LU factorization of the band matrix A, as
///     computed by `lapack::gbtrf`. U is stored as an upper triangular band
///     matrix with kl+ku superdiagonals in rows 1 to kl+ku+1, and
///     the multipliers used during the factorization are stored in
///     rows kl+ku+2 to 2*kl+ku+1.
///
/// @param[in] ldab
///     The leading dimension of the array AB. ldab >= 2*kl+ku+1.
///
/// @param[in] ipiv
///     The vector ipiv of length n.
///     The pivot indices; for 1 <= i <= n, row i of the matrix was
///     interchanged with row ipiv(i).
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
/// @ingroup gbsv_computational
int64_t gbcon(
    lapack::Norm norm, int64_t n, int64_t kl, int64_t ku,
    std::complex<double> const* AB, int64_t ldab,
    int64_t const* ipiv, double anorm,
    double* rcond )
{
    char norm_ = to_char( norm );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int ldab_ = to_lapack_int( ldab );
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
    lapack::vector< double > rwork( (n) );

    LAPACK_zgbcon(
        &norm_, &n_, &kl_, &ku_,
        (lapack_complex_double*) AB, &ldab_,
        ipiv_ptr, &anorm, rcond,
        (lapack_complex_double*) &work[0],
        &rwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
