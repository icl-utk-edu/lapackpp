// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>

#if defined(BLAS_HAVE_MKL) || defined(LAPACK_HAVE_MKL)
    #if (defined(BLAS_ILP64) || defined(LAPACK_ILP64)) && ! defined(MKL_ILP64)
        #define MKL_ILP64
    #endif
    #include <mkl_lapacke.h>
#else
    #include <lapacke.h>
#endif

int main()
{
    int n = 5;
    // symmetric positive definite A = L L^T, with exact L.
    // -1 values in upper triangle (viewed column-major) are not referenced.
    double A[] = {
        4,  2,  0,  0,  0,
       -1,  5,  2,  0,  0,
       -1, -1,  5,  2,  0,
       -1, -1, -1,  5,  2,
       -1, -1, -1, -1,  5
    };
    double L[] = {
         2,  1,  0,  0,  0,
        -1,  2,  1,  0,  0,
        -1, -1,  2,  1,  0,
        -1, -1, -1,  2,  1,
        -1, -1, -1, -1,  2
    };
    int info = LAPACKE_dpotrf( LAPACK_COL_MAJOR, 'l', n, A, n );
    bool okay = (info == 0);
    for (int i = 0; i < 5*5; ++i) {
        okay = okay && (A[i] == L[i]);
    }
    printf( "%s\n", okay ? "ok" : "failed" );
    return ! okay;
}
