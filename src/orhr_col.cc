// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"

#if LAPACK_VERSION >= 30900  // >= 3.9.0

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup unknown
int64_t orhr_col(
    int64_t m, int64_t n, int64_t nb,
    float* A, int64_t lda,
    float* T, int64_t ldt,
    float* D )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_error_if(m < n);
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int nb_ = (lapack_int) nb;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int info_ = 0;

    LAPACK_sorhr_col(
        &m_, &n_, &nb_,
        A, &lda_,
        T, &ldt_,
        D, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Takes an m-by-n real matrix Q_in with orthonormal columns
///  as input, stored in A, and performs Householder Reconstruction (HR),
///  i.e. reconstructs Householder vectors V(i) implicitly representing
///  another m-by-n matrix Q_out, with the property that Q_in = Q_out*S,
///  where S is an n-by-n diagonal matrix with diagonal entries
///  equal to +1 or -1. The Householder vectors (columns V(i) of V) are
///  stored in A on output, and the diagonal entries of S are stored in D.
///  Block reflectors are also returned in T
///  (same output format as `lapack::geqrt`).
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @since LAPACK 3.9.0
///
/// @param[in] m
///     The number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix A. m >= n >= 0.
///
/// @param[in] nb
///     The column block size to be used in the reconstruction
///     of Householder column vector blocks in the array A and
///     corresponding block reflectors in the array T. nb >= 1.
///     (Note that if nb > n, then n is used instead of nb
///     as the column block size.)
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     \n
///     On entry:
///     \n
///         The array A contains an m-by-n orthonormal matrix Q_in,
///         i.e the columns of A are orthogonal unit vectors.
///     \n
///     On exit:
///     \n
///         The elements below the diagonal of A represent the unit
///         lower-trapezoidal matrix V of Householder column vectors
///         V(i). The unit diagonal entries of V are not stored
///         (same format as the output below the diagonal in A from
///         `lapack::geqrt`). The matrix T and the matrix V stored on output
///         in A implicitly define Q_out.
///     \n
///         The elements above the diagonal contain the factor U
///         of the "modified" LU-decomposition:
///         Q_in - ( S ) = V * U
///         ( 0 )
///         where 0 is a (m-n)-by-(m-n) zero matrix.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @param[out] T
///     The max( 1, min(nb, n) )-by-n matrix T, stored in an ldt-by-n array.
///     dimension (ldt, n)
///     \n
///     Let NOCB = Number_of_output_col_blocks
///         = CEIL(n/nb)
///     \n
///     On exit, T(1:nb, 1:n) contains NOCB upper-triangular
///     block reflectors used to define Q_out stored in compact
///     form as a sequence of upper-triangular nb-by-nb column
///     blocks (same format as the output T in `lapack::geqrt`).
///     The matrix T and the matrix V stored on output in A
///     implicitly define Q_out. NOTE: The lower triangles
///     below the upper-triangular blocks will be filled with
///     zeros. See Further Details.
///
/// @param[in] ldt
///     The leading dimension of the array T.
///     ldt >= max(1,min(nb,n)).
///
/// @param[out] D
///     The vector D of length min(m,n).
///     The elements can be only plus or minus one.
///     \n
///     D(i) is constructed as D(i) = -SIGN(Q_in_i(i,i)), where
///     1 <= i <= min(m,n), and Q_in_i is Q_in after performing
///     i-1 steps of “modified” Gaussian elimination.
///     See Further Details.
///
/// @retval = 0: successful exit
///
/// @ingroup unknown
int64_t orhr_col(
    int64_t m, int64_t n, int64_t nb,
    double* A, int64_t lda,
    double* T, int64_t ldt,
    double* D )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int nb_ = (lapack_int) nb;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int info_ = 0;

    LAPACK_dorhr_col(
        &m_, &n_, &nb_,
        A, &lda_,
        T, &ldt_,
        D, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.9.0
