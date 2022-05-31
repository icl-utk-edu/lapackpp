// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>

#include "config.h"

#define LAPACK_dlagsy FORTRAN_NAME(dlagsy, DLAGSY)

#ifdef __cplusplus
extern "C"
#endif
void LAPACK_dlagsy(
    lapack_int const* n, lapack_int const* k,
    double const* d,
    double* a, lapack_int const* lda,
    lapack_int* iseed,
    double* work,
    lapack_int* info );

int main()
{
    const lapack_int n = 5, k = 5;
    lapack_int iseed[4] = { 0, 1, 2, 3 };
    double d[ n ] = { 1, 2, 3, 4, 5 };
    double A[ n*n ];
    double work[ 2*n ];
    lapack_int info = -1234;
    LAPACK_dlagsy( &n, &k, d, A, &n, iseed, work, &info );
    bool okay = (info == 0);
    printf( "%s\n", okay ? "ok" : "failed" );
    return ! okay;
}
