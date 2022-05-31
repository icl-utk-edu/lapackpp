// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas.hh"
#include "lapack.hh"
#include "error.hh"
//#include "check_ortho.hh"

#include <vector>

///-----------------------------------------------------------------------------
/// Checks error for over- and under-determined problems AX ~= B.
/// This works for the various least-squares routines.
///
/// In over-determined case,
/// checks if residual is orthogonal to colspan( op(A) ), saving error in result[0].
///
/// In under-determined case,
/// checks that X is in rowspan( op(A) ), saving error in result[0].
///
/// In consistent case (m <= n or B = A X0 for some X0),
/// checks that residual is small, saving error in result[1].
///
/// gels passes if result[0] < tol and result[1] < tol.
///
/// On entry, A, B are the original input data to gels, X is the output of gels.
/// A is m-by-n, op(A) is opAm-by-opAn, B is opAm-by-nrhs, X is opAn-by-nrhs.
///
template< typename scalar_t >
void check_gels(
    bool consistent,
    lapack::Op trans,
    int64_t m, int64_t n, int64_t nrhs,
    scalar_t const* A, int64_t lda,
    scalar_t const* X, int64_t ldx,
    scalar_t const* B, int64_t ldb,
    blas::real_type< scalar_t > result[2] )
{
    using real_t = blas::real_type<scalar_t>;
    using blas::Op;
    using blas::conj;

    result[0] = 0;
    result[1] = 0;

    int64_t opAm, opAn;
    lapack::Norm norm;
    if (trans == lapack::Op::NoTrans) {
        opAm = m;
        opAn = n;
        norm = lapack::Norm::One;
    }
    else {
        opAm = n;
        opAn = m;
        norm = lapack::Norm::Inf;
    }

    real_t opA_norm = lapack::lange( norm, m, n, A, lda );
    real_t   B_norm = lapack::lange( lapack::Norm::One, opAm, nrhs, B, ldb );
    real_t   X_norm = lapack::lange( lapack::Norm::One, opAn, nrhs, X, ldx );
    real_t error;

    // residual R = B - op(A) X
    std::vector< scalar_t > R( ldb * nrhs );
    lapack::lacpy( lapack::MatrixType::General, opAm, nrhs, B, ldb, &R[0], ldb );
    blas::gemm( blas::Layout::ColMajor, trans, Op::NoTrans, opAm, nrhs, opAn,
                -1.0, A, lda,
                      X, ldx,
                 1.0, &R[0], ldb );

    if (opAm >= opAn) {
        //--------------------------------------------------
        // Over-determined case, least squares solution.
        // Check that the residual R = AX - B is orthogonal to op(A):
        //
        //      || R^H op(A) ||_1
        //     ------------------------------------------- < tol * epsilon
        //      max(m, n, nrhs) || op(A) ||_1 * || B ||_1

        // todo: scale residual to unit max, and scale error below
        // see LAPACK [sdcz]qrt17.f
        //real_t R_max = slate::norm(slate::Norm::Max, B);
        //slate::scale(1, R_max, B);

        // X = R^H op(A)  (opAm-by-nrhs)^H (opAm-by-opAn) = (nrhs-by-opAm) (opAm-by-opAn)
        std::vector< scalar_t > RA( nrhs * opAn );
        blas::gemm( blas::Layout::ColMajor, Op::ConjTrans, trans, nrhs, opAn, opAm,
                    1.0, &R[0], ldb,
                         A, lda,
                    0.0, &RA[0], nrhs );

        // || R^H op(A) ||_1 == || X ||_inf
        error = lapack::lange( lapack::Norm::One, nrhs, opAn, &RA[0], nrhs );
        if (opA_norm != 0)
            error /= opA_norm;
        // todo: error *= R_max
        if (B_norm != 0)
            error /= B_norm;
        error /= blas::max(m, n, nrhs);
        result[0] = error;
    }
    else {
        //--------------------------------------------------
        // opAm < opAn
        // Under-determined case, minimum norm solution.
        // Check that X is in the row-span of op(A),
        // i.e., it has no component in the null space of op(A),
        // by doing QR factorization of D = [ op(A)^H, X ] and examining
        // E = R( opAm : opAm+nrhs-1, opAm : opAm+nrhs-1 ).
        //
        //      || E ||_max / max(m, n, nrhs) < tol * epsilon

        // op(A)^H is opAn-by-opAm, X is opAn-by-nrhs
        // D = [ op(A)^H, X ] is opAn-by-(opAm + nrhs)
        int64_t ldd = opAn;
        size_t size_D = ldd * (opAm + nrhs);
        size_t size_tau = blas::min( opAn, opAm + nrhs );
        std::vector< scalar_t > D( size_D );
        std::vector< scalar_t > tau( size_tau );

        if (trans == Op::NoTrans) {
            // copy op(A)^H = A^H -> D1
            // Alas! No transpose routine in LAPACK.
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = 0; i < m; ++i)
                    D[ j + i*ldd ] = conj( A[ i + j*lda ] );
        }
        else {
            // copy op(A)^H = A -> D1
            lapack::lacpy( lapack::MatrixType::General, m, n, A, lda, &D[0], ldd );
        }

        // copy X -> D2
        lapack::lacpy( lapack::MatrixType::General, opAn, nrhs, A, lda, &D[0], ldd );

        // QR of D
        int64_t info = lapack::geqrf( opAn, opAm + nrhs, &D[0], ldd, &tau[0] );
        require( info == 0 );

        // error = || R_{opAm : opAn-1, opAm : opAm+nrhs-1} ||_max
        error = lapack::lantr( lapack::Norm::Max, lapack::Uplo::Upper,
                               lapack::Diag::NonUnit, blas::min( opAn - opAm, nrhs ), nrhs,
                               &D[ opAm + opAm*ldd ], ldd );
        error /= blas::max(m, n, nrhs);
        result[0] = error;
    }

    //--------------------------------------------------
    // If op(A) X = B is consistent, because either B = op(A) X0
    // or opAm <= opAn, check the residual R:
    //
    //      || R ||_1
    //     ----------------------------------- < tol * epsilon
    //      max(m, n) || op(A) ||_1 || X ||_1
    if (consistent || opAm <= opAn) {
        error = lapack::lange( lapack::Norm::One, opAm, nrhs, &R[0], ldb );
        if (opA_norm != 0)
            error /= opA_norm;
        if (X_norm != 0)
            error /= X_norm;
        error /= blas::max(m, n);
        result[1] = error;
    }
}
