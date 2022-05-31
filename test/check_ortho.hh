// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas.hh"
#include "lapack.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// Computes error measure:
// || I - U^H U || / m  if rowcol == Col (cols are orthogonal; m >= n)
// or
// || I - U U^H || / n  if rowcol == Row (rows are orthogonal; m <= n)
// Similar to LAPACK testing zunt01
template< typename scalar_t >
blas::real_type< scalar_t > check_orthogonality(
    lapack::RowCol rowcol,
    int64_t m, int64_t n,
    scalar_t const* U, int64_t ldu )
{
    using namespace blas;
    using namespace lapack;
    using real_t = blas::real_type< scalar_t >;

    int64_t minmn = min( m, n );
    int64_t ldr = minmn;
    int64_t k;
    Op transU;
    if (rowcol == RowCol::Row) {
        if (m > n)
            throw lapack::Error( "rowcol == row && m > n", __func__ );
        transU = Op::NoTrans;
        k = n;
    }
    else {
        if (m < n)
            throw lapack::Error( "rowcol == col && m < n", __func__ );
        transU = Op::ConjTrans;
        k = m;
    }

    // R = I - U^H U (col) or I - U U^H (row)
    std::vector< scalar_t > R( minmn * minmn );
    laset( MatrixType::Upper, minmn, minmn, 0.0, 1.0, &R[0], ldr );
    herk( Layout::ColMajor, Uplo::Upper, transU, minmn, k, -1.0, U, ldu, 1.0, &R[0], ldr );

    // resid = || R || / k
    real_t resid = lanhe( Norm::One, Uplo::Upper, minmn, &R[0], ldr ) / k;
    return resid;
}
