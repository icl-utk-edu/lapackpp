// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup bdsvd
int64_t bdsqr(
    lapack::Uplo uplo, int64_t n, int64_t ncvt, int64_t nru, int64_t ncc,
    float* D,
    float* E,
    float* VT, int64_t ldvt,
    float* U, int64_t ldu,
    float* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ncvt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nru) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ncc) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int ncvt_ = (lapack_int) ncvt;
    lapack_int nru_ = (lapack_int) nru;
    lapack_int ncc_ = (lapack_int) ncc;
    lapack_int ldvt_ = (lapack_int) ldvt;
    lapack_int ldu_ = (lapack_int) ldu;
    lapack_int ldc_ = (lapack_int) ldc;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (4*n) );

    LAPACK_sbdsqr(
        &uplo_, &n_, &ncvt_, &nru_, &ncc_,
        D,
        E,
        VT, &ldvt_,
        U, &ldu_,
        C, &ldc_,
        &work[0], &info_
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
/// @ingroup bdsvd
int64_t bdsqr(
    lapack::Uplo uplo, int64_t n, int64_t ncvt, int64_t nru, int64_t ncc,
    double* D,
    double* E,
    double* VT, int64_t ldvt,
    double* U, int64_t ldu,
    double* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ncvt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nru) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ncc) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int ncvt_ = (lapack_int) ncvt;
    lapack_int nru_ = (lapack_int) nru;
    lapack_int ncc_ = (lapack_int) ncc;
    lapack_int ldvt_ = (lapack_int) ldvt;
    lapack_int ldu_ = (lapack_int) ldu;
    lapack_int ldc_ = (lapack_int) ldc;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (4*n) );

    LAPACK_dbdsqr(
        &uplo_, &n_, &ncvt_, &nru_, &ncc_,
        D,
        E,
        VT, &ldvt_,
        U, &ldu_,
        C, &ldc_,
        &work[0], &info_
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
/// @ingroup bdsvd
int64_t bdsqr(
    lapack::Uplo uplo, int64_t n, int64_t ncvt, int64_t nru, int64_t ncc,
    float* D,
    float* E,
    std::complex<float>* VT, int64_t ldvt,
    std::complex<float>* U, int64_t ldu,
    std::complex<float>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ncvt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nru) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ncc) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int ncvt_ = (lapack_int) ncvt;
    lapack_int nru_ = (lapack_int) nru;
    lapack_int ncc_ = (lapack_int) ncc;
    lapack_int ldvt_ = (lapack_int) ldvt;
    lapack_int ldu_ = (lapack_int) ldu;
    lapack_int ldc_ = (lapack_int) ldc;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > rwork( (4*n) );

    LAPACK_cbdsqr(
        &uplo_, &n_, &ncvt_, &nru_, &ncc_,
        D,
        E,
        (lapack_complex_float*) VT, &ldvt_,
        (lapack_complex_float*) U, &ldu_,
        (lapack_complex_float*) C, &ldc_,
        &rwork[0], &info_
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
/// Computes the singular values and, optionally, the right and/or
/// left singular vectors from the singular value decomposition (SVD) of
/// a real n-by-n (upper or lower) bidiagonal matrix B using the implicit
/// zero-shift QR algorithm. The SVD of B has the form
/// \[
///     B = Q S P^H
/// \]
/// where S is the diagonal matrix of singular values, Q is an orthogonal
/// matrix of left singular vectors, and P is an orthogonal matrix of
/// right singular vectors. If left singular vectors are requested, this
/// subroutine actually returns $U Q$ instead of Q, and, if right singular
/// vectors are requested, this subroutine returns $P^H V^H$ instead of
/// $P^H$, for given input matrices U and VT $= V^H$. When U and VT are
/// the unitary matrices that reduce a general matrix A to bidiagonal
/// form: $A = U B V^H$, as computed by `lapack::gebrd`, then
/// \[
///     A = (U Q) S (P^H V^H)
/// \]
/// is the SVD of A. Optionally, the subroutine may also compute $Q^H C$
/// for a given input matrix C.
///
/// See "Computing Small Singular Values of Bidiagonal Matrices With
/// Guaranteed High Relative Accuracy," by J. Demmel and W. Kahan,
/// LAPACK Working Note #3 (or SIAM J. Sci. Statist. Comput. vol. 11,
/// no. 5, pp. 873-912, Sept 1990) and
/// "Accurate singular values and differential qd algorithms," by
/// B. Parlett and V. Fernando, Technical Report CPAM-554, Mathematics
/// Department, University of California at Berkeley, July 1992
/// for a detailed description of the algorithm.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: B is upper bidiagonal;
///     - lapack::Uplo::Lower: B is lower bidiagonal.
///
/// @param[in] n
///     The order of the matrix B. n >= 0.
///
/// @param[in] ncvt
///     The number of columns of the matrix VT. ncvt >= 0.
///
/// @param[in] nru
///     The number of rows of the matrix U. nru >= 0.
///
/// @param[in] ncc
///     The number of columns of the matrix C. ncc >= 0.
///
/// @param[in,out] D
///     The vector D of length n.
///     On entry, the n diagonal elements of the bidiagonal matrix B.
///     On successful exit, the singular values of B in decreasing
///     order.
///
/// @param[in,out] E
///     The vector E of length n-1.
///     On entry, the n-1 offdiagonal elements of the bidiagonal
///     matrix B.
///     On successful exit, E is destroyed; if return value > 0, D and E
///     will contain the diagonal and superdiagonal elements of a
///     bidiagonal matrix orthogonally equivalent to the one given
///     as input.
///
/// @param[in,out] VT
///     The n-by-ncvt matrix VT, stored in an ldvt-by-ncvt array.
///     On entry, an n-by-ncvt matrix VT.
///     On exit, VT is overwritten by $P^H V^H$.
///     Not referenced if ncvt = 0.
///
/// @param[in] ldvt
///     The leading dimension of the array VT.
///     ldvt >= max(1,n) if ncvt > 0; ldvt >= 1 if ncvt = 0.
///
/// @param[in,out] U
///     The nru-by-n matrix U, stored in an ldu-by-n array.
///     On entry, an nru-by-n matrix U.
///     On exit, U is overwritten by $U Q$.
///     Not referenced if nru = 0.
///
/// @param[in] ldu
///     The leading dimension of the array U. ldu >= max(1,nru).
///
/// @param[in,out] C
///     The n-by-ncc matrix C, stored in an ldc-by-ncc array.
///     On entry, an n-by-ncc matrix C.
///     On exit, C is overwritten by $Q^H C$.
///     Not referenced if ncc = 0.
///
/// @param[in] ldc
///     The leading dimension of the array C.
///     ldc >= max(1,n) if ncc > 0; ldc >=1 if ncc = 0.
///
/// @return = 0: successful exit
/// @return > 0: the algorithm did not converge; D and E contain the
///              elements of a bidiagonal matrix which is orthogonally
///              similar to the input matrix B; if return value = i, i
///              elements of E have not converged to zero.
///
/// @ingroup bdsvd
int64_t bdsqr(
    lapack::Uplo uplo, int64_t n, int64_t ncvt, int64_t nru, int64_t ncc,
    double* D,
    double* E,
    std::complex<double>* VT, int64_t ldvt,
    std::complex<double>* U, int64_t ldu,
    std::complex<double>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ncvt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nru) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ncc) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int ncvt_ = (lapack_int) ncvt;
    lapack_int nru_ = (lapack_int) nru;
    lapack_int ncc_ = (lapack_int) ncc;
    lapack_int ldvt_ = (lapack_int) ldvt;
    lapack_int ldu_ = (lapack_int) ldu;
    lapack_int ldc_ = (lapack_int) ldc;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > rwork( (4*n) );

    LAPACK_zbdsqr(
        &uplo_, &n_, &ncvt_, &nru_, &ncc_,
        D,
        E,
        (lapack_complex_double*) VT, &ldvt_,
        (lapack_complex_double*) U, &ldu_,
        (lapack_complex_double*) C, &ldc_,
        &rwork[0], &info_
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
