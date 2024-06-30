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
/// @ingroup bdsvd
int64_t bdsdc(
    lapack::Uplo uplo, lapack::Job compq, int64_t n,
    float* D,
    float* E,
    float* U, int64_t ldu,
    float* VT, int64_t ldvt,
    float* Q,
    int64_t* IQ )
{
    char uplo_ = to_char( uplo );
    char compq_ = to_char_comp( compq );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldu_ = to_lapack_int( ldu );
    lapack_int ldvt_ = to_lapack_int( ldvt );
    lapack_int info_ = 0;

    // IQ disabled for now, due to complicated dimension
    lapack_int IQ_[1];
    lapack_int *IQ_ptr = &IQ_[0];

    // formulas from docs
    int64_t lwork = 0;
    switch (compq) {
        case Job::NoVec:      lwork = 4*n; break;
        case Job::Vec:        lwork = 6*n; break;
        case Job::CompactVec: lwork = 3*n*n + 4*n; break;
        default:
            assert( false );
            break;
    }

    // allocate workspace
    lapack::vector< float > work( (max( 1, lwork )) );
    lapack::vector< lapack_int > iwork( (8*n) );

    LAPACK_sbdsdc(
        &uplo_, &compq_, &n_,
        D,
        E,
        U, &ldu_,
        VT, &ldvt_,
        Q,
        IQ_ptr,
        &work[0],
        &iwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the singular value decomposition (SVD) of a real
/// n-by-n (upper or lower) bidiagonal matrix B: $B = U S V^T$,
/// using a divide and conquer method, where S is a diagonal matrix
/// with non-negative diagonal elements (the singular values of B), and
/// U and VT $= V^T$ are orthogonal matrices of left and right singular vectors,
/// respectively. `bdsdc` can be used to compute all singular values,
/// and optionally, singular vectors or singular vectors in compact form.
///
/// This code makes very mild assumptions about floating point
/// arithmetic. It will work on machines with a guard digit in
/// add/subtract, or on those binary machines without guard digits
/// which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
/// It could conceivably fail on hexadecimal or decimal machines
/// without guard digits, but we know of none. See `lapack::lasd3` for details.
///
/// The code currently calls `lapack::lasdq` if singular values only are desired.
/// However, it can be slightly modified to compute singular values
/// using the divide and conquer method.
///
/// Overloaded versions are available for
/// `float`, `double`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: B is upper bidiagonal.
///     - lapack::Uplo::Lower: B is lower bidiagonal.
///
/// @param[in] compq
///     Whether singular vectors are to be computed:
///     - lapack::Job::NoVec: Compute singular values only;
///     - lapack::Job::CompactVec: Compute singular values and compute singular
///             vectors in compact form
///             [this option not yet implemented in LAPACK++.];
///     - lapack::Job::Vec: Compute singular values and singular vectors.
///
/// @param[in] n
///     The order of the matrix B. n >= 0.
///
/// @param[in,out] D
///     The vector D of length n.
///     On entry, the n diagonal elements of the bidiagonal matrix B.
///     On successful exit, the singular values of B.
///
/// @param[in,out] E
///     The vector E of length n-1.
///     On entry, the elements of E contain the offdiagonal
///     elements of the bidiagonal matrix whose SVD is desired.
///     On exit, E has been destroyed.
///
/// @param[out] U
///     The n-by-n matrix U, stored in an ldu-by-n array.
///     If compq = Vec, then:
///     on successful exit, U contains the left singular vectors
///     of the bidiagonal matrix.
///     For other values of compq, U is not referenced.
///
/// @param[in] ldu
///     The leading dimension of the array U. ldu >= 1.
///     If singular vectors are desired, then ldu >= max( 1, n ).
///
/// @param[out] VT
///     The n-by-n matrix VT, stored in an ldvt-by-n array.
///     If compq = Vec, then:
///     on successful exit, VT^T contains the right singular
///     vectors of the bidiagonal matrix.
///     For other values of compq, VT is not referenced.
///
/// @param[in] ldvt
///     The leading dimension of the array VT. ldvt >= 1.
///     If singular vectors are desired, then ldvt >= max( 1, n ).
///
/// @param[out] Q
///     [This option not yet implemented in LAPACK++.]
///     If compq = CompactVec, then:
///     The vector Q of length ldq.
//
//      On successful exit, Q and IQ contain the left
//      and right singular vectors in a compact form,
//      requiring O(n log n) space instead of 2*n^2.
//      In particular, Q contains all the DOUBLE PRECISION data in
//      ldq >= n*(11 + 2*SMLSIZ + 8*INT(LOG_2(n/(SMLSIZ+1))))
//      words of memory, where SMLSIZ is returned by ILAENV and
//      is equal to the maximum size of the subproblems at the
//      bottom of the computation tree (usually about 25).
//
///     For other values of compq, Q is not referenced.
///
/// @param[out] IQ
///     [This option not yet implemented in LAPACK++.]
///     If compq = CompactVec, then:
///     The vector IQ of length ldiq.
//
//      On successful exit, Q and IQ contain the left
//      and right singular vectors in a compact form,
//      requiring O(n log n) space instead of 2*n^2.
//      In particular, IQ contains all INTEGER data in
//      LDIQ >= n*(3 + 3*INT(LOG_2(n/(SMLSIZ+1))))
//      words of memory, where SMLSIZ is returned by ILAENV and
//      is equal to the maximum size of the subproblems at the
//      bottom of the computation tree (usually about 25).
//
///     For other values of compq, IQ is not referenced.
///
/// @return = 0: successful exit.
/// @return > 0: The algorithm failed to compute a singular value.
///              The update process of divide and conquer failed.
///
/// @ingroup bdsvd
int64_t bdsdc(
    lapack::Uplo uplo, lapack::Job compq, int64_t n,
    double* D,
    double* E,
    double* U, int64_t ldu,
    double* VT, int64_t ldvt,
    double* Q,
    int64_t* IQ )
{
    char uplo_ = to_char( uplo );
    char compq_ = to_char_comp( compq );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldu_ = to_lapack_int( ldu );
    lapack_int ldvt_ = to_lapack_int( ldvt );
    lapack_int info_ = 0;

    // IQ disabled for now, due to complicated dimension
    lapack_int IQ_[1];
    lapack_int *IQ_ptr = &IQ_[0];

    // formulas from docs
    int64_t lwork = 0;
    switch (compq) {
        case Job::NoVec:      lwork = 4*n; break;
        case Job::Vec:        lwork = 6*n; break;
        case Job::CompactVec: lwork = 3*n*n + 4*n; break;
        default:
            assert( false );
            break;
    }

    // allocate workspace
    lapack::vector< double > work( (max( 1, lwork )) );
    lapack::vector< lapack_int > iwork( (8*n) );

    LAPACK_dbdsdc(
        &uplo_, &compq_, &n_,
        D,
        E,
        U, &ldu_,
        VT, &ldvt_,
        Q,
        IQ_ptr,
        &work[0],
        &iwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
