// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"

#if LAPACK_VERSION >= 30300  // >= v3.3

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
void lartgp(
    float f, float g,
    float* cs,
    float* sn,
    float* r )
{

    LAPACK_slartgp(
        &f, &g, cs, sn, r );
}

// -----------------------------------------------------------------------------
void lartgp(
    double f, double g,
    double* cs,
    double* sn,
    double* r )
{

    LAPACK_dlartgp(
        &f, &g, cs, sn, r );
}

}  // namespace lapack

#endif  // LAPACK >= v3.3
