// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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
    // lapacke.h uses LAPACK_ILP64; CMake passes only BLAS++'s BLAS_ILP64.
    #if defined(BLAS_ILP64) && ! defined(LAPACK_ILP64)
        #define LAPACK_ILP64
    #endif

    // OpenBLAS built with SYMBOLSUFFIX (e.g., 64_) also suffixes LAPACKE
    // functions, exporting LAPACKE_dpstrf64_. Defined before including
    // lapacke.h, this works whether or not the header's declarations are
    // suffixed. LAPACK_FORTRAN_SUFFIX takes precedence; BLAS_FORTRAN_SUFFIX
    // is the fallback.
    #if defined(LAPACK_FORTRAN_SUFFIX)
        #define LAPACKE_SUFFIX_ LAPACK_FORTRAN_SUFFIX
    #elif defined(BLAS_FORTRAN_SUFFIX)
        #define LAPACKE_SUFFIX_ BLAS_FORTRAN_SUFFIX
    #endif

    #ifdef LAPACKE_SUFFIX_
        #define LAPACKE_CONCAT_( a, b ) a##b
        #define LAPACKE_CONCAT(  a, b ) LAPACKE_CONCAT_( a, b )
        #define LAPACKE_dpstrf LAPACKE_CONCAT( LAPACKE_dpstrf, LAPACKE_SUFFIX_ )
    #endif

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
