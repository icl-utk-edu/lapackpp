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
int64_t gbtrf(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    float* AB, int64_t ldab,
    int64_t* ipiv )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int ldab_ = to_lapack_int( ldab );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( (min(m,n)) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    LAPACK_sgbtrf(
        &m_, &n_, &kl_, &ku_,
        AB, &ldab_,
        ipiv_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gbsv_computational
int64_t gbtrf(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    double* AB, int64_t ldab,
    int64_t* ipiv )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int ldab_ = to_lapack_int( ldab );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( (min(m,n)) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    LAPACK_dgbtrf(
        &m_, &n_, &kl_, &ku_,
        AB, &ldab_,
        ipiv_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gbsv_computational
int64_t gbtrf(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    std::complex<float>* AB, int64_t ldab,
    int64_t* ipiv )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int ldab_ = to_lapack_int( ldab );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( (min(m,n)) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    LAPACK_cgbtrf(
        &m_, &n_, &kl_, &ku_,
        (lapack_complex_float*) AB, &ldab_,
        ipiv_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes an LU factorization of an m-by-n band matrix A
/// using partial pivoting with row interchanges.
///
/// This is the blocked version of the algorithm, calling Level 3 BLAS.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] m
///     The number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix A. n >= 0.
///
/// @param[in] kl
///     The number of subdiagonals within the band of A. kl >= 0.
///
/// @param[in] ku
///     The number of superdiagonals within the band of A. ku >= 0.
///
/// @param[in,out] AB
///     The n-by-n band matrix AB, stored in an ldab-by-n array.
///     On entry, the matrix A in band storage, in rows kl+1 to
///     2*kl+ku+1; rows 1 to kl of the array need not be set.
///     The j-th column of A is stored in the j-th column of the
///     array AB as follows:
///     \n
///     AB(kl+ku+1+i-j,j) = A(i,j) for max(1,j-ku) <= i <= min(m,j+kl)
///     \n
///     On exit, details of the factorization: U is stored as an
///     upper triangular band matrix with kl+ku superdiagonals in
///     rows 1 to kl+ku+1, and the multipliers used during the
///     factorization are stored in rows kl+ku+2 to 2*kl+ku+1.
///     See below for further details.
///
/// @param[in] ldab
///     The leading dimension of the array AB. ldab >= 2*kl+ku+1.
///
/// @param[out] ipiv
///     The vector ipiv of length min(m,n).
///     The pivot indices; for 1 <= i <= min(m,n), row i of the
///     matrix was interchanged with row ipiv(i).
///
/// @return = 0: successful exit
/// @return > 0: if return value = i, U(i,i) is exactly zero. The factorization
///              has been completed, but the factor U is exactly
///              singular, and division by zero will occur if it is used
///              to solve a system of equations.
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The band storage scheme is illustrated by the following example, when
/// m = n = 6, kl = 2, ku = 1:
///
///     On entry:                        On exit:
///
///      *    *    *    +    +    +       *    *    *   u14  u25  u36
///      *    *    +    +    +    +       *    *   u13  u24  u35  u46
///      *   a12  a23  a34  a45  a56      *   u12  u23  u34  u45  u56
///     a11  a22  a33  a44  a55  a66     u11  u22  u33  u44  u55  u66
///     a21  a32  a43  a54  a65   *      m21  m32  m43  m54  m65   *
///     a31  a42  a53  a64   *    *      m31  m42  m53  m64   *    *
///
/// Array elements marked * are not used by the routine; elements marked
/// + need not be set on entry, but are required by the routine to store
/// elements of U because of fill-in resulting from the row interchanges.
///
/// @ingroup gbsv_computational
int64_t gbtrf(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    std::complex<double>* AB, int64_t ldab,
    int64_t* ipiv )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int ldab_ = to_lapack_int( ldab );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( (min(m,n)) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    LAPACK_zgbtrf(
        &m_, &n_, &kl_, &ku_,
        (lapack_complex_double*) AB, &ldab_,
        ipiv_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

}  // namespace lapack
