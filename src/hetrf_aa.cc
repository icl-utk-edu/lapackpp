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
/// @ingroup hesv_aa_computational
int64_t hetrf_aa(
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
    LAPACK_chetrf_aa(
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

    LAPACK_chetrf_aa(
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
/// Computes the factorization of a hermitian matrix A
/// using the Aasen's algorithm. The form of the factorization is
/// \[
///     A = U T U^H
/// \]
/// or
/// \[
///     A = L T L^H
/// \]
/// where U (or L) is a product of permutation and unit upper (lower)
/// triangular matrices, and T is a hermitian tridiagonal matrix.
///
/// This is the blocked version of the algorithm, calling Level 3 BLAS.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this is an alias for `lapack::sytrf_aa`.
/// For complex symmetric matrices, see `lapack::sytrf_aa`.
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
///     - On entry, the hermitian matrix A. If uplo = Upper, the leading
///     n-by-n upper triangular part of A contains the upper
///     triangular part of the matrix A, and the strictly lower
///     triangular part of A is not referenced. If uplo = Lower, the
///     leading n-by-n lower triangular part of A contains the lower
///     triangular part of the matrix A, and the strictly upper
///     triangular part of A is not referenced.
///     - On exit, the tridiagonal matrix is stored in the diagonals
///     and the subdiagonals of A just below (or above) the diagonals,
///     and L is stored below (or above) the subdiaonals, when uplo
///     is 'L' (or 'U').
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[out] ipiv
///     The vector ipiv of length n.
///     On exit, it contains the details of the interchanges, i.e.,
///     the row and column k of A were interchanged with the
///     row and column ipiv(k).
///
/// @return = 0: successful exit
///
/// @ingroup hesv_aa_computational
int64_t hetrf_aa(
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
    LAPACK_zhetrf_aa(
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

    LAPACK_zhetrf_aa(
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

#endif  // LAPACK >= 3.7
