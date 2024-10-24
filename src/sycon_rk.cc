// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#if LAPACK_VERSION >= 30700  // >= 3.7

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup sysv_rk_computational
int64_t sycon_rk(
    lapack::Uplo uplo, int64_t n,
    float const* A, int64_t lda,
    float const* E,
    int64_t const* ipiv, float anorm,
    float* rcond )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int const* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int const* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (2*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_ssycon_3(
        &uplo_, &n_,
        A, &lda_,
        E,
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
/// @ingroup sysv_rk_computational
int64_t sycon_rk(
    lapack::Uplo uplo, int64_t n,
    double const* A, int64_t lda,
    double const* E,
    int64_t const* ipiv, double anorm,
    double* rcond )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int const* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int const* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (2*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_dsycon_3(
        &uplo_, &n_,
        A, &lda_,
        E,
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
/// @ingroup sysv_rk_computational
int64_t sycon_rk(
    lapack::Uplo uplo, int64_t n,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* E,
    int64_t const* ipiv, float anorm,
    float* rcond )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
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

    LAPACK_csycon_3(
        &uplo_, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) E,
        ipiv_ptr, &anorm, rcond,
        (lapack_complex_float*) &work[0],
        &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Estimates the reciprocal of the condition number (in the
/// 1-norm) of a symmetric matrix A using the factorization
/// computed by `lapack::sytrf_rk`:
/// \[
///     A = P U D U^T P^T,
/// \]
/// or
/// \[
///     A = P L D L^T P^T,
/// \]
/// where U (or L) is unit upper (or lower) triangular matrix,
/// $U^T$ (or $L^T$) is the transpose of U (or L), P is a permutation
/// matrix, $P^T$ is the transpose of P, and D is symmetric and block
/// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
///
/// An estimate is obtained for $|| A^{-1} ||_1,$ and the reciprocal of the
/// condition number is computed as $\text{rcond} = 1 / (|| A ||_1 * || A^{-1} ||_1).$
/// This routine uses the BLAS-3 solver `lapack::hetrs_rk`.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, `lapack::hecon_rk` is an alias for this.
/// For complex Hermitian matrices, see `lapack::hecon_rk`.
///
/// @since LAPACK 3.7.0.
///
/// @param[in] uplo
///     Specifies whether the details of the factorization are
///     stored as an upper or lower triangular matrix:
///     - lapack::Uplo::Upper: Upper triangular, form is $A = P U D U^T P^T;$
///     - lapack::Uplo::Lower: Lower triangular, form is $A = P L D L^T P^T.$
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     Diagonal of the block diagonal matrix D and factors U or L
///     as computed by `lapack::sytrf_rk`:
///     - ONLY diagonal elements of the symmetric block diagonal
///         matrix D on the diagonal of A, i.e. D(k,k) = A(k,k);
///         (superdiagonal (or subdiagonal) elements of D
///         should be provided on entry in array E), and
///     - If uplo = Upper: factor U in the superdiagonal part of A.
///     - If uplo = Lower: factor L in the subdiagonal part of A.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[in] E
///     The vector E of length n.
///     On entry, contains the superdiagonal (or subdiagonal)
///     elements of the symmetric block diagonal matrix D
///     with 1-by-1 or 2-by-2 diagonal blocks, where
///     - If uplo = Upper: E(i) = D(i-1,i),i=2:n, E(1) not referenced;
///     - If uplo = Lower: E(i) = D(i+1,i),i=1:n-1, E(n) not referenced.
///
///     - Note: For 1-by-1 diagonal block D(k), where
///     1 <= k <= n, the element E(k) is not referenced in both
///     uplo = Upper or uplo = Lower cases.
///
/// @param[in] ipiv
///     The vector ipiv of length n.
///     Details of the interchanges and the block structure of D
///     as determined by `lapack::sytrf_rk`.
///
/// @param[in] anorm
///     The 1-norm of the original matrix A.
///
/// @param[out] rcond
///     The reciprocal of the condition number of the matrix A,
///     computed as rcond = 1/(anorm * ainv_norm), where ainv_norm is an
///     estimate of the 1-norm of $A^{-1}$ computed in this routine.
///
/// @return = 0: successful exit
///
/// @ingroup sysv_rk_computational
int64_t sycon_rk(
    lapack::Uplo uplo, int64_t n,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* E,
    int64_t const* ipiv, double anorm,
    double* rcond )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
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

    LAPACK_zsycon_3(
        &uplo_, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) E,
        ipiv_ptr, &anorm, rcond,
        (lapack_complex_double*) &work[0],
        &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.7
