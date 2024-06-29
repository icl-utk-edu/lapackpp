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
int64_t opmtr(
    lapack::Side side, lapack::Uplo uplo, lapack::Op trans, int64_t m, int64_t n,
    float const* AP,
    float const* tau,
    float* C, int64_t ldc )
{
    // for real, map ConjTrans to Trans
    if (trans == Op::ConjTrans)
        trans = Op::Trans;

    char side_ = to_char( side );
    char uplo_ = to_char( uplo );
    char trans_ = to_char( trans );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldc_ = to_lapack_int( ldc );
    lapack_int info_ = 0;

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    lapack::vector< float > work( lwork );

    LAPACK_sopmtr(
        &side_, &uplo_, &trans_, &m_, &n_,
        AP,
        tau,
        C, &ldc_,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t opmtr(
    lapack::Side side, lapack::Uplo uplo, lapack::Op trans, int64_t m, int64_t n,
    double const* AP,
    double const* tau,
    double* C, int64_t ldc )
{
    // for real, map ConjTrans to Trans
    if (trans == Op::ConjTrans)
        trans = Op::Trans;

    char side_ = to_char( side );
    char uplo_ = to_char( uplo );
    char trans_ = to_char( trans );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldc_ = to_lapack_int( ldc );
    lapack_int info_ = 0;

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    lapack::vector< double > work( lwork );

    LAPACK_dopmtr(
        &side_, &uplo_, &trans_, &m_, &n_,
        AP,
        tau,
        C, &ldc_,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
