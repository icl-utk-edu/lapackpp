// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SCALING_HH
#define SCALING_HH

#include <blas.hh>

// -----------------------------------------------------------------------------
/// Does column scaling:
///     A[:, j] *= Beta[j] for j = 0, ..., n-1.
/// A is m-by-n. Beta is n-length vector.
template <typename scalar_t, typename data_t>
void col_scale(
    int m, int n,
    data_t* A, int lda,
    scalar_t const* Beta )
{
    for (int j = 0; j < n; ++j) {
        blas::scal( m, Beta[ j ], &A[ j*lda ], 1 );
    }
}

// -----------------------------------------------------------------------------
/// Does row scaling:
///     A[i, :] *= Alpha[i] for i = 0, ..., m-1.
/// A is m-by-n. Alpha is m-length vector.
template <typename scalar_t, typename data_t>
void row_scale(
    int m, int n,
    data_t const* Alpha,
    scalar_t* A, int lda )
{
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) {
            A[ i + j*lda ] *= Alpha[ i ];
        }
    }
}

// -----------------------------------------------------------------------------
/// Does row and column scaling:
///     A[i, j] *= Alpha[i] * Beta[j] for i = 0, ..., m-1 and j = 0, ..., n-1.
/// A is m-by-n. Alpha is m-length vector. Beta is n-length vector.
template <typename scalar_t, typename data_t>
void row_col_scale(
    int m, int n,
    scalar_t const* Alpha,
    data_t* A, int lda,
    scalar_t const* Beta )
{
    for (int j = 0; j < m; ++j) {
        scalar_t beta_j = Beta[ j ];
        for (int i = 0; i < n; ++i) {
            A[ i + j*lda ] *= Alpha[ i ] * beta_j;
        }
    }
}

#endif  //  #ifndef SCALING_HH
