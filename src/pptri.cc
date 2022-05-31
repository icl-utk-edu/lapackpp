// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup ppsv_computational
int64_t pptri(
    lapack::Uplo uplo, int64_t n,
    float* AP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    LAPACK_spptri(
        &uplo_, &n_,
        AP, &info_
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
/// @ingroup ppsv_computational
int64_t pptri(
    lapack::Uplo uplo, int64_t n,
    double* AP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    LAPACK_dpptri(
        &uplo_, &n_,
        AP, &info_
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
/// @ingroup ppsv_computational
int64_t pptri(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* AP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    LAPACK_cpptri(
        &uplo_, &n_,
        (lapack_complex_float*) AP, &info_
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
/// Computes the inverse of a Hermitian positive definite
/// matrix A using the Cholesky factorization $A = U^H U$ or $A = L L^H$
/// computed by `lapack::pptrf`.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangular factor is stored in AP;
///     - lapack::Uplo::Lower: Lower triangular factor is stored in AP.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in,out] AP
///     The vector AP of length n*(n+1)/2.
///     - On entry, the triangular factor U or L from the Cholesky
///     factorization $A = U^H U$ or $A = L L^H,$ packed columnwise as
///     a linear array. The j-th column of U or L is stored in the
///     array AP as follows:
///       - if uplo = Upper, AP(i + (j-1)*j/2) = U(i,j) for 1 <= i <= j;
///       - if uplo = Lower, AP(i + (j-1)*(2n-j)/2) = L(i,j) for j <= i <= n.
///
///     - On exit, the upper or lower triangle of the (Hermitian)
///     inverse of A, overwriting the input factor U or L.
///
/// @return = 0: successful exit
/// @return > 0: if return value = i, the (i,i) element of the factor U or L is
///     zero, and the inverse could not be computed.
///
/// @ingroup ppsv_computational
int64_t pptri(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* AP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    LAPACK_zpptri(
        &uplo_, &n_,
        (lapack_complex_double*) AP, &info_
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
