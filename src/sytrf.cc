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
/// @ingroup sysv_computational
int64_t sytrf(
    lapack::Uplo uplo, int64_t n,
    float* A, int64_t lda,
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
    float qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_ssytrf(
        &uplo_, &n_,
        A, &lda_,
        ipiv_ptr,
        qry_work, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< float > work( lwork_ );

    LAPACK_ssytrf(
        &uplo_, &n_,
        A, &lda_,
        ipiv_ptr,
        &work[0], &lwork_, &info_
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
/// @ingroup sysv_computational
int64_t sytrf(
    lapack::Uplo uplo, int64_t n,
    double* A, int64_t lda,
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
    double qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_dsytrf(
        &uplo_, &n_,
        A, &lda_,
        ipiv_ptr,
        qry_work, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< double > work( lwork_ );

    LAPACK_dsytrf(
        &uplo_, &n_,
        A, &lda_,
        ipiv_ptr,
        &work[0], &lwork_, &info_
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
/// @ingroup sysv_computational
int64_t sytrf(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda,
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
    LAPACK_csytrf(
        &uplo_, &n_,
        (lapack_complex_float*) A, &lda_,
        ipiv_ptr,
        (lapack_complex_float*) qry_work, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );

    LAPACK_csytrf(
        &uplo_, &n_,
        (lapack_complex_float*) A, &lda_,
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
/// Computes the factorization of a symmetric matrix A
/// using the Bunch-Kaufman diagonal pivoting method. The form of the
/// factorization is
/// \[
///     A = U D U^T
/// \]
/// or
/// \[
///     A = L D L^T
/// \]
/// where U (or L) is a product of permutation and unit upper (lower)
/// triangular matrices, and D is symmetric and block diagonal with
/// 1-by-1 and 2-by-2 diagonal blocks.
///
/// This is the blocked version of the algorithm, calling Level 3 BLAS.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, `lapack::hetrf` is an alias for this.
/// For complex Hermitian matrices, see `lapack::hetrf`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangle of A is stored;
///     - lapack::Uplo::Lower: Lower triangle of A is stored.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     On entry, the symmetric matrix A.
///     - If uplo = Upper, the leading
///     n-by-n upper triangular part of A contains the upper
///     triangular part of the matrix A, and the strictly lower
///     triangular part of A is not referenced.
///
///     - If uplo = Lower, the
///     leading n-by-n lower triangular part of A contains the lower
///     triangular part of the matrix A, and the strictly upper
///     triangular part of A is not referenced.
///
///     - On exit, the block diagonal matrix D and the multipliers used
///     to obtain the factor U or L (see below for further details).
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[out] ipiv
///     The vector ipiv of length n.
///     Details of the interchanges and the block structure of D.
///     - If ipiv(k) > 0, then rows and columns k and ipiv(k) were
///     interchanged and D(k,k) is a 1-by-1 diagonal block.
///
///     - If uplo = Upper and ipiv(k) = ipiv(k-1) < 0,
///     then rows and columns k-1 and -ipiv(k) were interchanged
///     and D(k-1:k,k-1:k) is a 2-by-2 diagonal block.
///
///     - If uplo = Lower and ipiv(k) = ipiv(k+1) < 0,
///     then rows and columns k+1 and -ipiv(k) were interchanged
///     and D(k:k+1,k:k+1) is a 2-by-2 diagonal block.
///
/// @return = 0: successful exit
/// @return > 0: if return value = i,
///     D(i,i) is exactly zero. The factorization
///     has been completed, but the block diagonal matrix D is
///     exactly singular, and division by zero will occur if it
///     is used to solve a system of equations.
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// If uplo = Upper, then $A = U D U^T,$ where
/// \[
///     U = P(n) U(n) \dots P(k) U(k) \dots,
/// \]
/// i.e., U is a product of terms $P(k) U(k),$ where k decreases from n to
/// 1 in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1
/// and 2-by-2 diagonal blocks D(k). P(k) is a permutation matrix as
/// defined by ipiv(k), and U(k) is a unit upper triangular matrix, such
/// that if the diagonal block D(k) is of order s (s = 1 or 2), then
///
///             (   I    v    0   )   k-s
///     U(k) =  (   0    I    0   )   s
///             (   0    0    I   )   n-k
///                k-s   s   n-k
///
/// If s = 1, D(k) overwrites A(k,k), and v overwrites A(1:k-1,k).
/// If s = 2, the upper triangle of D(k) overwrites A(k-1,k-1), A(k-1,k),
/// and A(k,k), and v overwrites A(1:k-2,k-1:k).
///
/// If uplo = Lower, then $A = L D L^T,$ where
/// \[
///     L = P(1) L(1) \dots P(k) L(k) \dots,
/// \]
/// i.e., L is a product of terms $P(k) L(k),$ where k increases from 1 to
/// n in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1
/// and 2-by-2 diagonal blocks D(k). P(k) is a permutation matrix as
/// defined by ipiv(k), and L(k) is a unit lower triangular matrix, such
/// that if the diagonal block D(k) is of order s (s = 1 or 2), then
///
///             (   I    0     0   )  k-1
///     L(k) =  (   0    I     0   )  s
///             (   0    v     I   )  n-k-s+1
///                k-1   s  n-k-s+1
///
/// If s = 1, D(k) overwrites A(k,k), and v overwrites A(k+1:n,k).
/// If s = 2, the lower triangle of D(k) overwrites A(k,k), A(k+1,k),
/// and A(k+1,k+1), and v overwrites A(k+2:n,k:k+1).
///
/// @ingroup sysv_computational
int64_t sytrf(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda,
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
    LAPACK_zsytrf(
        &uplo_, &n_,
        (lapack_complex_double*) A, &lda_,
        ipiv_ptr,
        (lapack_complex_double*) qry_work, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );

    LAPACK_zsytrf(
        &uplo_, &n_,
        (lapack_complex_double*) A, &lda_,
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
