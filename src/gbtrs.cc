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
int64_t gbtrs(
    lapack::Op trans, int64_t n, int64_t kl, int64_t ku, int64_t nrhs,
    float const* AB, int64_t ldab,
    int64_t const* ipiv,
    float* B, int64_t ldb )
{
    char trans_ = to_char( trans );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldab_ = to_lapack_int( ldab );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int const* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int const* ipiv_ptr = ipiv;
    #endif
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_sgbtrs(
        &trans_, &n_, &kl_, &ku_, &nrhs_,
        AB, &ldab_,
        ipiv_ptr,
        B, &ldb_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gbsv_computational
int64_t gbtrs(
    lapack::Op trans, int64_t n, int64_t kl, int64_t ku, int64_t nrhs,
    double const* AB, int64_t ldab,
    int64_t const* ipiv,
    double* B, int64_t ldb )
{
    char trans_ = to_char( trans );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldab_ = to_lapack_int( ldab );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int const* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int const* ipiv_ptr = ipiv;
    #endif
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_dgbtrs(
        &trans_, &n_, &kl_, &ku_, &nrhs_,
        AB, &ldab_,
        ipiv_ptr,
        B, &ldb_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gbsv_computational
int64_t gbtrs(
    lapack::Op trans, int64_t n, int64_t kl, int64_t ku, int64_t nrhs,
    std::complex<float> const* AB, int64_t ldab,
    int64_t const* ipiv,
    std::complex<float>* B, int64_t ldb )
{
    char trans_ = to_char( trans );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldab_ = to_lapack_int( ldab );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int const* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int const* ipiv_ptr = ipiv;
    #endif
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_cgbtrs(
        &trans_, &n_, &kl_, &ku_, &nrhs_,
        (lapack_complex_float*) AB, &ldab_,
        ipiv_ptr,
        (lapack_complex_float*) B, &ldb_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Solves a system of linear equations
/// \[
///     A   X = B,
/// \]
/// \[
///     A^T X = B,
/// \]
/// or
/// \[
///     A^H X = B
/// \]
/// with a general band matrix A using the LU factorization computed
/// by `lapack::gbtrf`.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] trans
///     The form of the system of equations.
///     - lapack::Op::NoTrans:   $A   X = B$ (No transpose)
///     - lapack::Op::Trans:     $A^T X = B$ (Transpose)
///     - lapack::Op::ConjTrans: $A^H X = B$ (Conjugate transpose)
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
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrix B. nrhs >= 0.
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
/// @param[in,out] B
///     The n-by-nrhs matrix B, stored in an ldb-by-nrhs array.
///     On entry, the right hand side matrix B.
///     On exit, the solution matrix X.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,n).
///
/// @return = 0: successful exit
///
/// @ingroup gbsv_computational
int64_t gbtrs(
    lapack::Op trans, int64_t n, int64_t kl, int64_t ku, int64_t nrhs,
    std::complex<double> const* AB, int64_t ldab,
    int64_t const* ipiv,
    std::complex<double>* B, int64_t ldb )
{
    char trans_ = to_char( trans );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int nrhs_ = to_lapack_int( nrhs );
    lapack_int ldab_ = to_lapack_int( ldab );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int const* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int const* ipiv_ptr = ipiv;
    #endif
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    LAPACK_zgbtrs(
        &trans_, &n_, &kl_, &ku_, &nrhs_,
        (lapack_complex_double*) AB, &ldab_,
        ipiv_ptr,
        (lapack_complex_double*) B, &ldb_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
