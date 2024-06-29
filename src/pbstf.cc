// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup pbsv_computational
int64_t pbstf(
    lapack::Uplo uplo, int64_t n, int64_t kd,
    float* AB, int64_t ldab )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int info_ = 0;

    LAPACK_spbstf(
        &uplo_, &n_, &kd_,
        AB, &ldab_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup pbsv_computational
int64_t pbstf(
    lapack::Uplo uplo, int64_t n, int64_t kd,
    double* AB, int64_t ldab )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int info_ = 0;

    LAPACK_dpbstf(
        &uplo_, &n_, &kd_,
        AB, &ldab_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup pbsv_computational
int64_t pbstf(
    lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<float>* AB, int64_t ldab )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int info_ = 0;

    LAPACK_cpbstf(
        &uplo_, &n_, &kd_,
        (lapack_complex_float*) AB, &ldab_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes a split Cholesky factorization of a
/// Hermitian positive definite band matrix A.
///
/// This routine is designed to be used in conjunction with `lapack::hbgst`.
///
/// The factorization has the form $A = S^H S$ where S is a band matrix
/// of the same bandwidth as A and the following structure:
/// \[
///     S = \begin{bmatrix}
///             U  &
///         \\  M  &  L
///     \end{bmatrix},
/// \]
/// where U is upper triangular of order m = (n+kd)/2, and L is lower
/// triangular of order n-m.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangle of A is stored;
///     - lapack::Uplo::Lower: Lower triangle of A is stored.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] kd
///     - If uplo = Upper, the number of superdiagonals of the matrix A;
///     - if uplo = Lower, the number of subdiagonals.
///     - kd >= 0.
///
/// @param[in,out] AB
///     The n-by-n band matrix AB, stored in an ldab-by-n array.
///     - On entry, the upper or lower triangle of the Hermitian band
///     matrix A, stored in the first kd+1 rows of the array. The
///     j-th column of A is stored in the j-th column of the array AB
///     as follows:
///       - if uplo = Upper, AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd) <= i <= j;
///       - if uplo = Lower, AB(1+i-j,j) = A(i,j) for j <= i <= min(n,j+kd).
///
///     - On successful exit, the factor S from the split Cholesky
///     factorization $A = S^H S.$ See Further Details.
///
/// @param[in] ldab
///     The leading dimension of the array AB. ldab >= kd+1.
///
/// @return = 0: successful exit
/// @return > 0: if return value = i, the factorization could not be completed,
///     because the updated element a(i,i) was negative; the
///     matrix A is not positive definite.
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The band storage scheme is illustrated by the following example, when
/// n = 7, kd = 2:
///
///     S = [ s11  s12  s13                     ]
///         [      s22  s23  s24                ]
///         [           s33  s34                ]
///         [                s44                ]
///         [           s53  s54  s55           ]
///         [                s64  s65  s66      ]
///         [                     s75  s76  s77 ]
///
/// If uplo = Upper, the array AB holds:
///
///     on entry:
///
///     [  *    *   a13  a24  a35  a46  a57 ]
///     [  *   a12  a23  a34  a45  a56  a67 ]
///     [ a11  a22  a33  a44  a55  a66  a77 ]
///
///     on exit:
///
///     [  *    *   s13  s24  s53^H  s64^H  s75^H ]
///     [  *   s12  s23  s34  s54^H  s65^H  s76^H ]
///     [ s11  s22  s33  s44  s55    s66    s77   ]
///
/// If uplo = Lower, the array AB holds:
///
///     on entry:
///
///     a11  a22  a33  a44  a55  a66  a77
///     a21  a32  a43  a54  a65  a76   *
///     a31  a42  a53  a64  a64   *    *
///
///     on exit:
///
///     s11    s22    s33    s44  s55  s66  s77
///     s12^H  s23^H  s34^H  s54  s65  s76   *
///     s13^H  s24^H  s53    s64  s75   *    *
///
/// Array elements marked * are not used by the routine; s12^H denotes
/// conj(s12); the diagonal elements of S are real.
///
/// @ingroup pbsv_computational
int64_t pbstf(
    lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<double>* AB, int64_t ldab )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int kd_ = to_lapack_int( kd );
    lapack_int ldab_ = to_lapack_int( ldab );
    lapack_int info_ = 0;

    LAPACK_zpbstf(
        &uplo_, &n_, &kd_,
        (lapack_complex_double*) AB, &ldab_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
