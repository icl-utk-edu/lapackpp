// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>

#include "config.h"

#define LAPACK_dposvxx FORTRAN_NAME(dposvxx, DPOSVXX)

#ifdef __cplusplus
extern "C"
#endif
void LAPACK_dposvxx(
    char const* fact, char const* uplo,
    lapack_int const* n, lapack_int const* nrhs,
    double* a, lapack_int const* lda,
    double* af, lapack_int const* ldaf,
    char* equed, double* s,
    double* b, lapack_int const* ldb,
    double* x, lapack_int const* ldx,
    double* rcond, double* rpvgrw, double* berr,
    lapack_int const* n_err_bnds, double* err_bnds_norm, double* err_bnds_comp,
    lapack_int const* nparams, double* params,
    double* work, lapack_int* iwork,
    lapack_int* info );

int main()
{
    const lapack_int n = 5, nrhs = 1, n_err_bnds = 3, nparams = 3;
    // symmetric positive definite
    double A[ n*n ] = {
        4, 1, 0, 0, 0,
        1, 4, 1, 0, 0,
        0, 1, 4, 1, 0,
        0, 0, 1, 4, 1,
        0, 0, 0, 1, 4
    };
    double AF[ n*n ];
    double B[ n*nrhs ] = { 1, 2, 3, 4, 5 };
    double X[ n*nrhs ] = { 1, 2, 3, 4, 5 };
    double S[ n ], rcond, rpivotgrowth, berr[ nrhs ];
    double err_bnds_norm[ nrhs*n_err_bnds ], err_bnds_comp[ nrhs*n_err_bnds ];
    double params[ nparams ] = { -1, -1, -1 };
    double work[ 4*n ];
    char equed = 'n';
    lapack_int iwork[ n ];
    lapack_int info = -1234;
    LAPACK_dposvxx( "n", "lower", &n, &nrhs,
                    A, &n, AF, &n,
                    &equed, S,
                    B, &n, X, &n,
                    &rcond, &rpivotgrowth, berr,
                    &n_err_bnds, err_bnds_norm, err_bnds_comp,
                    &nparams, params,
                    work, iwork,
                    &info );
    bool okay = (info == 0);
    printf( "%s\n", okay ? "ok" : "failed" );
    return ! okay;
}
