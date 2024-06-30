// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t disna(
    lapack::JobCond jobcond, int64_t m, int64_t n,
    float const* D,
    float* SEP )
{
    char jobcond_ = to_char( jobcond );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_sdisna(
        &jobcond_, &m_, &n_,
        D,
        SEP, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t disna(
    lapack::JobCond jobcond, int64_t m, int64_t n,
    double const* D,
    double* SEP )
{
    char jobcond_ = to_char( jobcond );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int info_ = 0;

    LAPACK_ddisna(
        &jobcond_, &m_, &n_,
        D,
        SEP, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
