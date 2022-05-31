// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas.hh"
#include "lapack.hh"
#include "error.hh"
#include "check_ortho.hh"

#include <vector>

// -----------------------------------------------------------------------------
// Computes error measures:
// result[0] = || A - U diag(S) VT || / (||A|| max(m,n)),
//                                    if jobu  != NoVec and jobvt != NoVec.
// result[1] = || I - U^H U || / m,   if jobu  != NoVec.
// result[2] = || I - VT VT^H || / n, if jobvt != NoVec.
// result[3] = 0 if S has non-negative values in non-increasing order, else 1.
template< typename scalar_t >
void check_svd(
    lapack::Job jobu, lapack::Job jobvt,
    int64_t m, int64_t n,
    scalar_t const* A, int64_t lda,
    blas::real_type< scalar_t > const* s,
    scalar_t const* U,  int64_t ldu,
    scalar_t const* VT, int64_t ldvt,
    blas::real_type< scalar_t > result[4] )
{
    using namespace blas;
    using namespace lapack;
    using real_t = blas::real_type< scalar_t >;

    if (jobu == Job::NoVec) {
        U = nullptr;
    }
    if (jobvt == Job::NoVec) {
        VT = nullptr;
    }
    int64_t minmn = min( m, n );
    int64_t maxmn = max( m, n );

    if (U != nullptr && VT != nullptr) {
        // check || A - U diag(S) VT || / (||A|| max(m,n))
        // R = A
        std::vector< scalar_t > R( m * n );
        lacpy( MatrixType::General, m, n, A, lda, &R[0], m );

        if (m >= n) {
            // SVT = diag(S) * VT (row scaling)
            std::vector< scalar_t > SVT( n * n );
            lacpy( MatrixType::General, n, n, VT, ldvt, &SVT[0], n );
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < n; ++i) {
                    SVT[ i + j*n ] *= s[i];
                }
            }
            // R = A - U * (SVT)
            gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, minmn /* == n */,
                  -1.0, U, ldu,
                        &SVT[0], n,
                   1.0, &R[0], m );
        }
        else {
            // US = U * diag(S) (col scaling)
            std::vector< scalar_t > US( m * m );
            lacpy( MatrixType::General, m, m, U, ldu, &US[0], m );
            for (int64_t j = 0; j < m; ++j) {
                scal( m, s[j], &US[j*m], 1 );
            }
            // R = A - (US) * VT
            gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, minmn /* == m */,
                  -1.0, &US[0], m,
                        VT, ldvt,
                   1.0, &R[0], m );
        }
        real_t eps = std::numeric_limits< real_t >::epsilon();
        real_t Anorm = lange( Norm::One, m, n, A, lda );
        real_t resid = lange( Norm::One, m, n, &R[0], m );
        if (Anorm >= resid) {
            resid = resid / Anorm / maxmn;
        }
        else if (Anorm > 0) {
            if (Anorm < 1)
                resid = min( resid, maxmn * Anorm ) / Anorm / maxmn;
            else
                resid = min( resid / Anorm, maxmn ) / maxmn;
        }
        else { // Anorm == 0
            resid = 1 / eps;
        }
        result[0] = resid;
    }

    if (U != nullptr) {
        // check || I - U^H U || / m
        int64_t ucols = (jobu == Job::AllVec ? m : minmn);
        result[1] = check_orthogonality( RowCol::Col, m, ucols, U, ldu );
    }

    if (VT != nullptr) {
        // check || I - VT VT^H || / n
        int64_t vrows = (jobvt == Job::AllVec ? n : minmn);
        result[2] = check_orthogonality( RowCol::Row, vrows, n, VT, ldvt );
    }

    // check s >= 0 and s is non-increasing
    result[3] = 0;
    for (int64_t i = 0; i < minmn; ++i) {
        if (s[i] < 0 || (i < minmn - 1 && s[i] < s[i+1]))
            result[3] = 1;
    }
}
