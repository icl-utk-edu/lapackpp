// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack/fortran.h"

namespace lapack {

// -----------------------------------------------------------------------------
void lartg(
    float f, float g,
    float* cs,
    float* sn,
    float* r )
{

    LAPACK_slartg(
        &f, &g, cs, sn, r );
}

// -----------------------------------------------------------------------------
void lartg(
    double f, double g,
    double* cs,
    double* sn,
    double* r )
{

    LAPACK_dlartg(
        &f, &g, cs, sn, r );
}

// -----------------------------------------------------------------------------
void lartg(
    std::complex<float> f, std::complex<float> g,
    float* cs,
    std::complex<float>* sn,
    std::complex<float>* r )
{

    LAPACK_clartg(
        (lapack_complex_float*) &f,
        (lapack_complex_float*) &g,
        cs,
        (lapack_complex_float*) sn,
        (lapack_complex_float*) r );
}

// -----------------------------------------------------------------------------
void lartg(
    std::complex<double> f, std::complex<double> g,
    double* cs,
    std::complex<double>* sn,
    std::complex<double>* r )
{

    LAPACK_zlartg(
        (lapack_complex_double*) &f,
        (lapack_complex_double*) &g,
        cs,
        (lapack_complex_double*) sn,
        (lapack_complex_double*) r );
}

}  // namespace lapack
