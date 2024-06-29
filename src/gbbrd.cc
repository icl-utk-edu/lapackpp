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
/// @ingroup gesvd_computational
int64_t gbbrd(
    lapack::Vect vect, int64_t m, int64_t n, int64_t ncc, int64_t kl, int64_t ku,
    float* AB, int64_t ldab,
    float* D,
    float* E,
    float* Q, int64_t ldq,
    float* PT, int64_t ldpt,
    float* C, int64_t ldc )
{
    char vect_ = to_char( vect );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ncc_ = to_lapack_int( ncc );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int ldpt_ = to_lapack_int( ldpt );
    lapack_int ldc_ = to_lapack_int( ldc );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (2*max(m,n)) );

    LAPACK_sgbbrd(
        &vect_, &m_, &n_, &ncc_, &kl_, &ku_,
        AB, &ldab_,
        D,
        E,
        Q, &ldq_,
        PT, &ldpt_,
        C, &ldc_,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesvd_computational
int64_t gbbrd(
    lapack::Vect vect, int64_t m, int64_t n, int64_t ncc, int64_t kl, int64_t ku,
    double* AB, int64_t ldab,
    double* D,
    double* E,
    double* Q, int64_t ldq,
    double* PT, int64_t ldpt,
    double* C, int64_t ldc )
{
    char vect_ = to_char( vect );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ncc_ = to_lapack_int( ncc );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int ldpt_ = to_lapack_int( ldpt );
    lapack_int ldc_ = to_lapack_int( ldc );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (2*max(m,n)) );

    LAPACK_dgbbrd(
        &vect_, &m_, &n_, &ncc_, &kl_, &ku_,
        AB, &ldab_,
        D,
        E,
        Q, &ldq_,
        PT, &ldpt_,
        C, &ldc_,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesvd_computational
int64_t gbbrd(
    lapack::Vect vect, int64_t m, int64_t n, int64_t ncc, int64_t kl, int64_t ku,
    std::complex<float>* AB, int64_t ldab,
    float* D,
    float* E,
    std::complex<float>* Q, int64_t ldq,
    std::complex<float>* PT, int64_t ldpt,
    std::complex<float>* C, int64_t ldc )
{
    char vect_ = to_char( vect );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ncc_ = to_lapack_int( ncc );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int ldpt_ = to_lapack_int( ldpt );
    lapack_int ldc_ = to_lapack_int( ldc );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (max(m,n)) );
    lapack::vector< float > rwork( (max(m,n)) );

    LAPACK_cgbbrd(
        &vect_, &m_, &n_, &ncc_, &kl_, &ku_,
        (lapack_complex_float*) AB, &ldab_,
        D,
        E,
        (lapack_complex_float*) Q, &ldq_,
        (lapack_complex_float*) PT, &ldpt_,
        (lapack_complex_float*) C, &ldc_,
        (lapack_complex_float*) &work[0],
        &rwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Reduces a general m-by-n band matrix A to real upper
/// bidiagonal form B by a unitary transformation: $Q^H A P = B$.
///
/// The routine computes B, and optionally forms $Q$ or $P^H$, or computes
/// $Q^H C$ for a given matrix C.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] vect
///     Whether or not the matrices Q and P^H are to be
///     formed.
///     - lapack::Vect::None: do not form $Q$ or $P^H$;
///     - lapack::Vect::Q:    form $Q$   only;
///     - lapack::Vect::P:    form $P^H$ only;
///     - lapack::Vect::Both: form both.
///
/// @param[in] m
///     The number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix A. n >= 0.
///
/// @param[in] ncc
///     The number of columns of the matrix C. ncc >= 0.
///
/// @param[in] kl
///     The number of subdiagonals of the matrix A. kl >= 0.
///
/// @param[in] ku
///     The number of superdiagonals of the matrix A. ku >= 0.
///
/// @param[in,out] AB
///     The m-by-n band matrix AB, stored in an ldab-by-n array.
///     On entry, the m-by-n band matrix A, stored in rows 1 to
///     kl+ku+1. The j-th column of A is stored in the j-th column of
///     the array AB as follows:
///     AB(ku+1+i-j,j) = A(i,j) for max(1,j-ku) <= i <= min(m,j+kl).
///     On exit, A is overwritten by values generated during the
///     reduction.
///
/// @param[in] ldab
///     The leading dimension of the array A. ldab >= kl+ku+1.
///
/// @param[out] D
///     The vector D of length min(m,n).
///     The diagonal elements of the bidiagonal matrix B.
///
/// @param[out] E
///     The vector E of length min(m,n)-1.
///     The superdiagonal elements of the bidiagonal matrix B.
///
/// @param[out] Q
///     The m-by-m matrix Q, stored in an ldq-by-m array.
///     - If vect = Q or Both, the m-by-m unitary matrix Q.
///     - If vect = None or P, the array Q is not referenced.
///
/// @param[in] ldq
///     The leading dimension of the array Q.
///     - If vect = Q or Both, ldq >= max(1,m);
///     - otherwise, ldq >= 1.
///
/// @param[out] PT
///     The n-by-n matrix PT, stored in an ldpt-by-n array.
///     - If vect = P or Both, the n-by-n unitary matrix $P^H$;
///     - If vect = None or Q, the array PT is not referenced.
///
/// @param[in] ldpt
///     The leading dimension of the array PT.
///     - If vect = P or Both, ldpt >= max(1,n);
///     - otherwise, ldpt >= 1.
///
/// @param[in,out] C
///     The m-by-ncc matrix C, stored in an ldc-by-ncc array.
///     On entry, an m-by-ncc matrix C.
///     On exit, C is overwritten by $Q^H C$.
///     C is not referenced if ncc = 0.
///
/// @param[in] ldc
///     The leading dimension of the array C.
///     - If ncc > 0, ldc >= max(1,m);
///     - if ncc = 0, ldc >= 1.
///
/// @return = 0: successful exit.
///
/// @ingroup gesvd_computational
int64_t gbbrd(
    lapack::Vect vect, int64_t m, int64_t n, int64_t ncc, int64_t kl, int64_t ku,
    std::complex<double>* AB, int64_t ldab,
    double* D,
    double* E,
    std::complex<double>* Q, int64_t ldq,
    std::complex<double>* PT, int64_t ldpt,
    std::complex<double>* C, int64_t ldc )
{
    char vect_ = to_char( vect );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ncc_ = to_lapack_int( ncc );
    lapack_int kl_ = to_lapack_int( kl );
    lapack_int ku_ = to_lapack_int( ku );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int ldpt_ = to_lapack_int( ldpt );
    lapack_int ldc_ = to_lapack_int( ldc );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (max(m,n)) );
    lapack::vector< double > rwork( (max(m,n)) );

    LAPACK_zgbbrd(
        &vect_, &m_, &n_, &ncc_, &kl_, &ku_,
        (lapack_complex_double*) AB, &ldab_,
        D,
        E,
        (lapack_complex_double*) Q, &ldq_,
        (lapack_complex_double*) PT, &ldpt_,
        (lapack_complex_double*) C, &ldc_,
        (lapack_complex_double*) &work[0],
        &rwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
