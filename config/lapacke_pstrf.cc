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
    // symmetric positive definite A = L L^T.
    // -1 values in upper triangle (viewed column-major) are not referenced.
    double A[] = {
        4,  2,  0,  0,  0,
       -1,  5,  2,  0,  0,
       -1, -1,  5,  2,  0,
       -1, -1, -1,  5,  2,
       -1, -1, -1, -1,  5
    };
    lapack_int ipiv[5] = { -1, -1, -1, -1, -1 };
    lapack_int rank = -1;
    double tol = -1;
    // With pivoting in pstrf, P^T A P = L2 L2^T.
    // Don't have exact L2 for comparison.
    lapack_int info = LAPACKE_dpstrf( LAPACK_COL_MAJOR, 'l', n, A, n, ipiv, &rank, tol );
    bool okay = (info == 0) && (rank == 5);
    printf( "%s\n", okay ? "ok" : "failed" );
    return ! okay;
}
