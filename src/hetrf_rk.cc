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
/// @ingroup hesv_rk_computational
int64_t hetrf_rk(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* E,
    int64_t* ipiv )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( (n) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_chetrf_rk(
        &uplo_, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) E,
        ipiv_ptr,
        (lapack_complex_float*) qry_work, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );

    LAPACK_chetrf_rk(
        &uplo_, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) E,
        ipiv_ptr,
        (lapack_complex_float*) &work[0], &lwork_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
// this is here just to get a doxygen entry
/// @see lapack::hetrf_rk
/// @ingroup hesv_rk_computational
#define hetrf_3 hetrf_rk

// -----------------------------------------------------------------------------
/// Computes the factorization of a Hermitian matrix A
/// using the bounded Bunch-Kaufman (rook) diagonal pivoting method:
/// \[
///     A = P U D U^H P^T
/// \]
/// or
/// \[
///     A = P L D L^H P^T,
/// \]
/// where U (or L) is unit upper (or lower) triangular matrix,
/// $U^H$ (or $L^H$) is the conjugate of U (or L), P is a permutation
/// matrix, $P^T$ is the transpose of P, and D is Hermitian and block
/// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
///
/// This is the blocked version of the algorithm, calling Level 3 BLAS.
/// For more information see Further Details section.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this is an alias for `lapack::sytrf_rk`.
/// For complex symmetric matrices, see `lapack::sytrf_rk`.
///
/// @since LAPACK 3.7.0.
/// This interface replaces the older `lapack::hetrf_rook`.
///
/// @param[in] uplo
///     Whether the upper or lower triangular part of the
///     Hermitian matrix A is stored:
///     - lapack::Uplo::Upper: Upper triangular
///     - lapack::Uplo::Lower: Lower triangular
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     - On entry, the Hermitian matrix A.
///       - If uplo = Upper: the leading n-by-n upper triangular part
///       of A contains the upper triangular part of the matrix A,
///       and the strictly lower triangular part of A is not
///       referenced.
///
///       - If uplo = Lower: the leading n-by-n lower triangular part
///       of A contains the lower triangular part of the matrix A,
///       and the strictly upper triangular part of A is not
///       referenced.
///
///     - On exit, contains:
///       - ONLY diagonal elements of the Hermitian block diagonal
///         matrix D on the diagonal of A, i.e. D(k,k) = A(k,k);
///         (superdiagonal (or subdiagonal) elements of D
///         are stored on exit in array E), and
///       - If uplo = Upper: factor U in the superdiagonal part of A.
///       - If uplo = Lower: factor L in the subdiagonal part of A.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[out] E
///     The vector E of length n.
///     On exit, contains the superdiagonal (or subdiagonal)
///     elements of the Hermitian block diagonal matrix D
///     with 1-by-1 or 2-by-2 diagonal blocks, where
///     - If uplo = Upper: E(i) = D(i-1,i), i=2:n, E(1) is set to 0;
///     - If uplo = Lower: E(i) = D(i+1,i), i=1:n-1, E(n) is set to 0.
///
///     - Note: For 1-by-1 diagonal block D(k), where
///     1 <= k <= n, the element E(k) is set to 0 in both
///     uplo = Upper or uplo = Lower cases.
///
/// @param[out] ipiv
///     The vector ipiv of length n.
///     ipiv describes the permutation matrix P in the factorization
///     of matrix A as follows. The absolute value of ipiv(k)
///     represents the index of row and column that were
///     interchanged with the k-th row and column. The value of uplo
///     describes the order in which the interchanges were applied.
///     Also, the sign of ipiv represents the block structure of
///     the Hermitian block diagonal matrix D with 1-by-1 or 2-by-2
///     diagonal blocks which correspond to 1 or 2 interchanges
///     at each factorization step. For more info see Further
///     Details section.
///     - If uplo = Upper,
///     (in factorization order, k decreases from n to 1):
///       a) A single positive entry ipiv(k) > 0 means:
///         D(k,k) is a 1-by-1 diagonal block.
///         If ipiv(k) != k, rows and columns k and ipiv(k) were
///         interchanged in the matrix A(1:n,1:n);
///         If ipiv(k) = k, no interchange occurred.
///
///       b) A pair of consecutive negative entries
///         ipiv(k) < 0 and ipiv(k-1) < 0 means:
///         D(k-1:k,k-1:k) is a 2-by-2 diagonal block.
///         (NOTE: negative entries in ipiv appear ONLY in pairs).
///         1) If -ipiv(k) != k, rows and columns
///         k and -ipiv(k) were interchanged
///         in the matrix A(1:n,1:n).
///         If -ipiv(k) = k, no interchange occurred.
///         2) If -ipiv(k-1) != k-1, rows and columns
///         k-1 and -ipiv(k-1) were interchanged
///         in the matrix A(1:n,1:n).
///         If -ipiv(k-1) = k-1, no interchange occurred.
///
///       c) In both cases a) and b), always ABS( ipiv(k) ) <= k.
///
///       d) NOTE: Any entry ipiv(k) is always NONZERO on output.
///
///     - If uplo = Lower,
///     (in factorization order, k increases from 1 to n):
///       a) A single positive entry ipiv(k) > 0 means:
///         D(k,k) is a 1-by-1 diagonal block.
///         If ipiv(k) != k, rows and columns k and ipiv(k) were
///         interchanged in the matrix A(1:n,1:n).
///         If ipiv(k) = k, no interchange occurred.
///
///       b) A pair of consecutive negative entries
///         ipiv(k) < 0 and ipiv(k+1) < 0 means:
///         D(k:k+1,k:k+1) is a 2-by-2 diagonal block.
///         (NOTE: negative entries in ipiv appear ONLY in pairs).
///         1) If -ipiv(k) != k, rows and columns
///         k and -ipiv(k) were interchanged
///         in the matrix A(1:n,1:n).
///         If -ipiv(k) = k, no interchange occurred.
///         2) If -ipiv(k+1) != k+1, rows and columns
///         k-1 and -ipiv(k-1) were interchanged
///         in the matrix A(1:n,1:n).
///         If -ipiv(k+1) = k+1, no interchange occurred.
///
///       c) In both cases a) and b), always ABS( ipiv(k) ) >= k.
///
///       d) NOTE: Any entry ipiv(k) is always NONZERO on output.
///
/// @return = 0: successful exit
/// @return > 0: If return value = i, the matrix A is singular, because:
///     If uplo = Upper: column i in the upper
///     triangular part of A contains all zeros.
///     If uplo = Lower: column i in the lower
///     triangular part of A contains all zeros.
///
///     Therefore D(i,i) is exactly zero, and superdiagonal
///     elements of column i of U (or subdiagonal elements of
///     column i of L ) are all zeros. The factorization has
///     been completed, but the block diagonal matrix D is
///     exactly singular, and division by zero will occur if
///     it is used to solve a system of equations.
///
///     NOTE: info only stores the first occurrence of
///     a singularity, any subsequent occurrence of singularity
///     is not stored in info even though the factorization
///     always completes.
///
/// @ingroup hesv_rk_computational
int64_t hetrf_rk(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* E,
    int64_t* ipiv )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ipiv_( (n) );
        lapack_int* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int* ipiv_ptr = ipiv;
    #endif
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_zhetrf_rk(
        &uplo_, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) E,
        ipiv_ptr,
        (lapack_complex_double*) qry_work, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );

    LAPACK_zhetrf_rk(
        &uplo_, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) E,
        ipiv_ptr,
        (lapack_complex_double*) &work[0], &lwork_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    #ifndef LAPACK_ILP64
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.7
