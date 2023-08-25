// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef ICL_LAPACKE_WRAPPERS_HH
#define ICL_LAPACKE_WRAPPERS_HH

#include "lapack/util.hh"
#include "lapack/config.h"  // may be newer than what's in lapacke.h

#include <complex>
#include <vector>

#ifdef BLAS_HAVE_MKL
    // define lapack_complex and MKL_Complex to be consistent
    #define MKL_Complex8  lapack_complex_float
    #define MKL_Complex16 lapack_complex_double
    #if (defined(BLAS_ILP64) || defined(LAPACK_ILP64)) && ! defined(MKL_ILP64)
        #define MKL_ILP64
    #endif
    #include <mkl_lapacke.h>
#else
    #include <lapacke.h>
#endif

// *after* lapacke.h, which may define LAPACK_GLOBAL macro
#include "lapack/mangling.h"

// This is in alphabetical order.

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gbcon(
    char norm, lapack_int n, lapack_int kl, lapack_int ku,
    float* AB, lapack_int ldab,
    lapack_int* ipiv, float anorm,
    float* rcond )
{
    return LAPACKE_sgbcon(
        LAPACK_COL_MAJOR, norm, n, kl, ku,
        AB, ldab,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_gbcon(
    char norm, lapack_int n, lapack_int kl, lapack_int ku,
    double* AB, lapack_int ldab,
    lapack_int* ipiv, double anorm,
    double* rcond )
{
    return LAPACKE_dgbcon(
        LAPACK_COL_MAJOR, norm, n, kl, ku,
        AB, ldab,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_gbcon(
    char norm, lapack_int n, lapack_int kl, lapack_int ku,
    std::complex<float>* AB, lapack_int ldab,
    lapack_int* ipiv, float anorm,
    float* rcond )
{
    return LAPACKE_cgbcon(
        LAPACK_COL_MAJOR, norm, n, kl, ku,
        (lapack_complex_float*) AB, ldab,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_gbcon(
    char norm, lapack_int n, lapack_int kl, lapack_int ku,
    std::complex<double>* AB, lapack_int ldab,
    lapack_int* ipiv, double anorm,
    double* rcond )
{
    return LAPACKE_zgbcon(
        LAPACK_COL_MAJOR, norm, n, kl, ku,
        (lapack_complex_double*) AB, ldab,
        ipiv, anorm,
        rcond );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gbequ(
    lapack_int m, lapack_int n, lapack_int kl, lapack_int ku,
    float* AB, lapack_int ldab,
    float* R,
    float* C,
    float* rowcnd,
    float* colcnd,
    float* amax )
{
    return LAPACKE_sgbequ(
        LAPACK_COL_MAJOR, m, n, kl, ku,
        AB, ldab,
        R,
        C,
        rowcnd,
        colcnd,
        amax );
}

inline lapack_int LAPACKE_gbequ(
    lapack_int m, lapack_int n, lapack_int kl, lapack_int ku,
    double* AB, lapack_int ldab,
    double* R,
    double* C,
    double* rowcnd,
    double* colcnd,
    double* amax )
{
    return LAPACKE_dgbequ(
        LAPACK_COL_MAJOR, m, n, kl, ku,
        AB, ldab,
        R,
        C,
        rowcnd,
        colcnd,
        amax );
}

inline lapack_int LAPACKE_gbequ(
    lapack_int m, lapack_int n, lapack_int kl, lapack_int ku,
    std::complex<float>* AB, lapack_int ldab,
    float* R,
    float* C,
    float* rowcnd,
    float* colcnd,
    float* amax )
{
    return LAPACKE_cgbequ(
        LAPACK_COL_MAJOR, m, n, kl, ku,
        (lapack_complex_float*) AB, ldab,
        R,
        C,
        rowcnd,
        colcnd,
        amax );
}

inline lapack_int LAPACKE_gbequ(
    lapack_int m, lapack_int n, lapack_int kl, lapack_int ku,
    std::complex<double>* AB, lapack_int ldab,
    double* R,
    double* C,
    double* rowcnd,
    double* colcnd,
    double* amax )
{
    return LAPACKE_zgbequ(
        LAPACK_COL_MAJOR, m, n, kl, ku,
        (lapack_complex_double*) AB, ldab,
        R,
        C,
        rowcnd,
        colcnd,
        amax );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gbrfs(
    char trans, lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs,
    float* AB, lapack_int ldab,
    float* AFB, lapack_int ldafb,
    lapack_int* ipiv,
    float* B, lapack_int ldb,
    float* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_sgbrfs(
        LAPACK_COL_MAJOR, trans, n, kl, ku, nrhs,
        AB, ldab,
        AFB, ldafb,
        ipiv,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_gbrfs(
    char trans, lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs,
    double* AB, lapack_int ldab,
    double* AFB, lapack_int ldafb,
    lapack_int* ipiv,
    double* B, lapack_int ldb,
    double* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_dgbrfs(
        LAPACK_COL_MAJOR, trans, n, kl, ku, nrhs,
        AB, ldab,
        AFB, ldafb,
        ipiv,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_gbrfs(
    char trans, lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs,
    std::complex<float>* AB, lapack_int ldab,
    std::complex<float>* AFB, lapack_int ldafb,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_cgbrfs(
        LAPACK_COL_MAJOR, trans, n, kl, ku, nrhs,
        (lapack_complex_float*) AB, ldab,
        (lapack_complex_float*) AFB, ldafb,
        ipiv,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_gbrfs(
    char trans, lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs,
    std::complex<double>* AB, lapack_int ldab,
    std::complex<double>* AFB, lapack_int ldafb,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_zgbrfs(
        LAPACK_COL_MAJOR, trans, n, kl, ku, nrhs,
        (lapack_complex_double*) AB, ldab,
        (lapack_complex_double*) AFB, ldafb,
        ipiv,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) X, ldx,
        ferr,
        berr );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gbsv(
    lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs,
    float* AB, lapack_int ldab,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_sgbsv(
        LAPACK_COL_MAJOR, n, kl, ku, nrhs,
        AB, ldab,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_gbsv(
    lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs,
    double* AB, lapack_int ldab,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dgbsv(
        LAPACK_COL_MAJOR, n, kl, ku, nrhs,
        AB, ldab,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_gbsv(
    lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs,
    std::complex<float>* AB, lapack_int ldab,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cgbsv(
        LAPACK_COL_MAJOR, n, kl, ku, nrhs,
        (lapack_complex_float*) AB, ldab,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_gbsv(
    lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs,
    std::complex<double>* AB, lapack_int ldab,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zgbsv(
        LAPACK_COL_MAJOR, n, kl, ku, nrhs,
        (lapack_complex_double*) AB, ldab,
        ipiv,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gbtrf(
    lapack_int m, lapack_int n, lapack_int kl, lapack_int ku,
    float* AB, lapack_int ldab,
    lapack_int* ipiv )
{
    return LAPACKE_sgbtrf(
        LAPACK_COL_MAJOR, m, n, kl, ku,
        AB, ldab,
        ipiv );
}

inline lapack_int LAPACKE_gbtrf(
    lapack_int m, lapack_int n, lapack_int kl, lapack_int ku,
    double* AB, lapack_int ldab,
    lapack_int* ipiv )
{
    return LAPACKE_dgbtrf(
        LAPACK_COL_MAJOR, m, n, kl, ku,
        AB, ldab,
        ipiv );
}

inline lapack_int LAPACKE_gbtrf(
    lapack_int m, lapack_int n, lapack_int kl, lapack_int ku,
    std::complex<float>* AB, lapack_int ldab,
    lapack_int* ipiv )
{
    return LAPACKE_cgbtrf(
        LAPACK_COL_MAJOR, m, n, kl, ku,
        (lapack_complex_float*) AB, ldab,
        ipiv );
}

inline lapack_int LAPACKE_gbtrf(
    lapack_int m, lapack_int n, lapack_int kl, lapack_int ku,
    std::complex<double>* AB, lapack_int ldab,
    lapack_int* ipiv )
{
    return LAPACKE_zgbtrf(
        LAPACK_COL_MAJOR, m, n, kl, ku,
        (lapack_complex_double*) AB, ldab,
        ipiv );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gbtrs(
    char trans, lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs,
    float* AB, lapack_int ldab,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_sgbtrs(
        LAPACK_COL_MAJOR, trans, n, kl, ku, nrhs,
        AB, ldab,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_gbtrs(
    char trans, lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs,
    double* AB, lapack_int ldab,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dgbtrs(
        LAPACK_COL_MAJOR, trans, n, kl, ku, nrhs,
        AB, ldab,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_gbtrs(
    char trans, lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs,
    std::complex<float>* AB, lapack_int ldab,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cgbtrs(
        LAPACK_COL_MAJOR, trans, n, kl, ku, nrhs,
        (lapack_complex_float*) AB, ldab,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_gbtrs(
    char trans, lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs,
    std::complex<double>* AB, lapack_int ldab,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zgbtrs(
        LAPACK_COL_MAJOR, trans, n, kl, ku, nrhs,
        (lapack_complex_double*) AB, ldab,
        ipiv,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gecon(
    char norm, lapack_int n,
    float* A, lapack_int lda, float anorm,
    float* rcond )
{
    return LAPACKE_sgecon(
        LAPACK_COL_MAJOR, norm, n,
        A, lda, anorm,
        rcond );
}

inline lapack_int LAPACKE_gecon(
    char norm, lapack_int n,
    double* A, lapack_int lda, double anorm,
    double* rcond )
{
    return LAPACKE_dgecon(
        LAPACK_COL_MAJOR, norm, n,
        A, lda, anorm,
        rcond );
}

inline lapack_int LAPACKE_gecon(
    char norm, lapack_int n,
    std::complex<float>* A, lapack_int lda, float anorm,
    float* rcond )
{
    return LAPACKE_cgecon(
        LAPACK_COL_MAJOR, norm, n,
        (lapack_complex_float*) A, lda, anorm,
        rcond );
}

inline lapack_int LAPACKE_gecon(
    char norm, lapack_int n,
    std::complex<double>* A, lapack_int lda, double anorm,
    double* rcond )
{
    return LAPACKE_zgecon(
        LAPACK_COL_MAJOR, norm, n,
        (lapack_complex_double*) A, lda, anorm,
        rcond );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_geequ(
    lapack_int m, lapack_int n,
    float* A, lapack_int lda,
    float* R,
    float* C,
    float* rowcnd,
    float* colcnd,
    float* amax )
{
    return LAPACKE_sgeequ(
        LAPACK_COL_MAJOR, m, n,
        A, lda,
        R,
        C,
        rowcnd,
        colcnd,
        amax );
}

inline lapack_int LAPACKE_geequ(
    lapack_int m, lapack_int n,
    double* A, lapack_int lda,
    double* R,
    double* C,
    double* rowcnd,
    double* colcnd,
    double* amax )
{
    return LAPACKE_dgeequ(
        LAPACK_COL_MAJOR, m, n,
        A, lda,
        R,
        C,
        rowcnd,
        colcnd,
        amax );
}

inline lapack_int LAPACKE_geequ(
    lapack_int m, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    float* R,
    float* C,
    float* rowcnd,
    float* colcnd,
    float* amax )
{
    return LAPACKE_cgeequ(
        LAPACK_COL_MAJOR, m, n,
        (lapack_complex_float*) A, lda,
        R,
        C,
        rowcnd,
        colcnd,
        amax );
}

inline lapack_int LAPACKE_geequ(
    lapack_int m, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    double* R,
    double* C,
    double* rowcnd,
    double* colcnd,
    double* amax )
{
    return LAPACKE_zgeequ(
        LAPACK_COL_MAJOR, m, n,
        (lapack_complex_double*) A, lda,
        R,
        C,
        rowcnd,
        colcnd,
        amax );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_geev(
    char jobvl, char jobvr, lapack_int n,
    float* A, lapack_int lda,
    std::complex<float>* W,
    float* VL, lapack_int ldvl,
    float* VR, lapack_int ldvr )
{
    std::vector< float > WR( n ), WI( n );
    lapack_int err = LAPACKE_sgeev(
        LAPACK_COL_MAJOR, jobvl, jobvr, n,
        A, lda,
        &WR[0], &WI[0],
        VL, ldvl,
        VR, ldvr );
    for (int64_t i = 0; i < n; ++i) {
        W[i] = std::complex<float>( WR[i], WI[i] );
    }
    return err;
}

inline lapack_int LAPACKE_geev(
    char jobvl, char jobvr, lapack_int n, double* A, lapack_int lda,
    std::complex<double>* W,
    double* VL, lapack_int ldvl,
    double* VR, lapack_int ldvr )
{
    std::vector< double > WR( n ), WI( n );
    lapack_int err = LAPACKE_dgeev(
        LAPACK_COL_MAJOR, jobvl, jobvr, n,
        A, lda,
        &WR[0], &WI[0],
        VL, ldvl,
        VR, ldvr );
    for (int64_t i = 0; i < n; ++i) {
        W[i] = std::complex<double>( WR[i], WI[i] );
    }
    return err;
}

inline lapack_int LAPACKE_geev(
    char jobvl, char jobvr, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* W,
    std::complex<float>* VL, lapack_int ldvl,
    std::complex<float>* VR, lapack_int ldvr )
{
    return LAPACKE_cgeev(
        LAPACK_COL_MAJOR, jobvl, jobvr, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) W,
        (lapack_complex_float*) VL, ldvl,
        (lapack_complex_float*) VR, ldvr );
}

inline lapack_int LAPACKE_geev(
    char jobvl, char jobvr, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* W,
    std::complex<double>* VL, lapack_int ldvl,
    std::complex<double>* VR, lapack_int ldvr )
{
    return LAPACKE_zgeev(
        LAPACK_COL_MAJOR, jobvl, jobvr, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) W,
        (lapack_complex_double*) VL, ldvl,
        (lapack_complex_double*) VR, ldvr );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gehrd(
    lapack_int n, lapack_int ilo, lapack_int ihi,
    float* A, lapack_int lda,
    float* tau )
{
    return LAPACKE_sgehrd(
        LAPACK_COL_MAJOR, n, ilo, ihi,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_gehrd(
    lapack_int n, lapack_int ilo, lapack_int ihi,
    double* A, lapack_int lda,
    double* tau )
{
    return LAPACKE_dgehrd(
        LAPACK_COL_MAJOR, n, ilo, ihi,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_gehrd(
    lapack_int n, lapack_int ilo, lapack_int ihi,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* tau )
{
    return LAPACKE_cgehrd(
        LAPACK_COL_MAJOR, n, ilo, ihi,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) tau );
}

inline lapack_int LAPACKE_gehrd(
    lapack_int n, lapack_int ilo, lapack_int ihi,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* tau )
{
    return LAPACKE_zgehrd(
        LAPACK_COL_MAJOR, n, ilo, ihi,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) tau );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gelqf(
    lapack_int m, lapack_int n,
    float* A, lapack_int lda,
    float* tau )
{
    return LAPACKE_sgelqf(
        LAPACK_COL_MAJOR, m, n,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_gelqf(
    lapack_int m, lapack_int n,
    double* A, lapack_int lda,
    double* tau )
{
    return LAPACKE_dgelqf(
        LAPACK_COL_MAJOR, m, n,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_gelqf(
    lapack_int m, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* tau )
{
    return LAPACKE_cgelqf(
        LAPACK_COL_MAJOR, m, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) tau );
}

inline lapack_int LAPACKE_gelqf(
    lapack_int m, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* tau )
{
    return LAPACKE_zgelqf(
        LAPACK_COL_MAJOR, m, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) tau );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gels(
    char trans, lapack_int m, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    float* B, lapack_int ldb )
{
    if (trans == 'C')
        trans = 'T';
    return LAPACKE_sgels(
        LAPACK_COL_MAJOR, trans, m, n, nrhs,
        A, lda,
        B, ldb );
}

inline lapack_int LAPACKE_gels(
    char trans, lapack_int m, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    double* B, lapack_int ldb )
{
    if (trans == 'C')
        trans = 'T';
    return LAPACKE_dgels(
        LAPACK_COL_MAJOR, trans, m, n, nrhs,
        A, lda,
        B, ldb );
}

inline lapack_int LAPACKE_gels(
    char trans, lapack_int m, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cgels(
        LAPACK_COL_MAJOR, trans, m, n, nrhs,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_gels(
    char trans, lapack_int m, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zgels(
        LAPACK_COL_MAJOR, trans, m, n, nrhs,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gelsd(
    lapack_int m, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    float* S, float rcond, lapack_int* rank )
{
    return LAPACKE_sgelsd(
        LAPACK_COL_MAJOR, m, n, nrhs,
        A, lda,
        B, ldb,
        S, rcond,
        rank );
}

inline lapack_int LAPACKE_gelsd(
    lapack_int m, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    double* S, double rcond, lapack_int* rank )
{
    return LAPACKE_dgelsd(
        LAPACK_COL_MAJOR, m, n, nrhs,
        A, lda,
        B, ldb,
        S, rcond,
        rank );
}

inline lapack_int LAPACKE_gelsd(
    lapack_int m, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    float* S, float rcond, lapack_int* rank )
{
    return LAPACKE_cgelsd(
        LAPACK_COL_MAJOR, m, n, nrhs,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb,
        S, rcond,
        rank );
}

inline lapack_int LAPACKE_gelsd(
    lapack_int m, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    double* S, double rcond, lapack_int* rank )
{
    return LAPACKE_zgelsd(
        LAPACK_COL_MAJOR, m, n, nrhs,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb,
        S, rcond,
        rank );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gelss(
    lapack_int m, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    float* S, float rcond, lapack_int* rank )
{
    return LAPACKE_sgelss(
        LAPACK_COL_MAJOR, m, n, nrhs,
        A, lda,
        B, ldb,
        S, rcond,
        rank );
}

inline lapack_int LAPACKE_gelss(
    lapack_int m, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    double* S, double rcond, lapack_int* rank )
{
    return LAPACKE_dgelss(
        LAPACK_COL_MAJOR, m, n, nrhs,
        A, lda,
        B, ldb,
        S, rcond,
        rank );
}

inline lapack_int LAPACKE_gelss(
    lapack_int m, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    float* S, float rcond, lapack_int* rank )
{
    return LAPACKE_cgelss(
        LAPACK_COL_MAJOR, m, n, nrhs,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb,
        S, rcond,
        rank );
}

inline lapack_int LAPACKE_gelss(
    lapack_int m, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    double* S, double rcond, lapack_int* rank )
{
    return LAPACKE_zgelss(
        LAPACK_COL_MAJOR, m, n, nrhs,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb,
        S, rcond,
        rank );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gelsy(
    lapack_int m, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    lapack_int* jpvt, float rcond, lapack_int* rank )
{
    return LAPACKE_sgelsy(
        LAPACK_COL_MAJOR, m, n, nrhs,
        A, lda,
        B, ldb,
        jpvt, rcond,
        rank );
}

inline lapack_int LAPACKE_gelsy(
    lapack_int m, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    lapack_int* jpvt, double rcond, lapack_int* rank )
{
    return LAPACKE_dgelsy(
        LAPACK_COL_MAJOR, m, n, nrhs,
        A, lda,
        B, ldb,
        jpvt, rcond,
        rank );
}

inline lapack_int LAPACKE_gelsy(
    lapack_int m, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    lapack_int* jpvt, float rcond, lapack_int* rank )
{
    return LAPACKE_cgelsy(
        LAPACK_COL_MAJOR, m, n, nrhs,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb,
        jpvt, rcond,
        rank );
}

inline lapack_int LAPACKE_gelsy(
    lapack_int m, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    lapack_int* jpvt, double rcond, lapack_int* rank )
{
    return LAPACKE_zgelsy(
        LAPACK_COL_MAJOR, m, n, nrhs,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb,
        jpvt, rcond,
        rank );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_geqlf(
    lapack_int m, lapack_int n,
    float* A, lapack_int lda,
    float* tau )
{
    return LAPACKE_sgeqlf(
        LAPACK_COL_MAJOR, m, n,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_geqlf(
    lapack_int m, lapack_int n,
    double* A, lapack_int lda,
    double* tau )
{
    return LAPACKE_dgeqlf(
        LAPACK_COL_MAJOR, m, n,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_geqlf(
    lapack_int m, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* tau )
{
    return LAPACKE_cgeqlf(
        LAPACK_COL_MAJOR, m, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) tau );
}

inline lapack_int LAPACKE_geqlf(
    lapack_int m, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* tau )
{
    return LAPACKE_zgeqlf(
        LAPACK_COL_MAJOR, m, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) tau );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_geqr(
    lapack_int m, lapack_int n,
    float* A, lapack_int lda,
    float* T, lapack_int tsize )
{
    return LAPACKE_sgeqr(
        LAPACK_COL_MAJOR, m, n,
        A, lda,
        T, tsize );
}

inline lapack_int LAPACKE_geqr(
    lapack_int m, lapack_int n,
    double* A, lapack_int lda,
    double* T, lapack_int tsize )
{
    return LAPACKE_dgeqr(
        LAPACK_COL_MAJOR, m, n,
        A, lda,
        T, tsize );
}

inline lapack_int LAPACKE_geqr(
    lapack_int m, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* T, lapack_int tsize )
{
    return LAPACKE_cgeqr(
        LAPACK_COL_MAJOR, m, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) T, tsize );
}

inline lapack_int LAPACKE_geqr(
    lapack_int m, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* T, lapack_int tsize )
{
    return LAPACKE_zgeqr(
        LAPACK_COL_MAJOR, m, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) T, tsize );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_geqrf(
    lapack_int m, lapack_int n,
    float* A, lapack_int lda,
    float* tau )
{
    return LAPACKE_sgeqrf(
        LAPACK_COL_MAJOR, m, n,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_geqrf(
    lapack_int m, lapack_int n,
    double* A, lapack_int lda,
    double* tau )
{
    return LAPACKE_dgeqrf(
        LAPACK_COL_MAJOR, m, n,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_geqrf(
    lapack_int m, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* tau )
{
    return LAPACKE_cgeqrf(
        LAPACK_COL_MAJOR, m, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) tau );
}

inline lapack_int LAPACKE_geqrf(
    lapack_int m, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* tau )
{
    return LAPACKE_zgeqrf(
        LAPACK_COL_MAJOR, m, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) tau );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gerfs(
    char trans, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    float* AF, lapack_int ldaf,
    lapack_int* ipiv,
    float* B, lapack_int ldb,
    float* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_sgerfs(
        LAPACK_COL_MAJOR, trans, n, nrhs,
        A, lda,
        AF, ldaf,
        ipiv,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_gerfs(
    char trans, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    double* AF, lapack_int ldaf,
    lapack_int* ipiv,
    double* B, lapack_int ldb,
    double* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_dgerfs(
        LAPACK_COL_MAJOR, trans, n, nrhs,
        A, lda,
        AF, ldaf,
        ipiv,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_gerfs(
    char trans, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* AF, lapack_int ldaf,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_cgerfs(
        LAPACK_COL_MAJOR, trans, n, nrhs,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) AF, ldaf,
        ipiv,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_gerfs(
    char trans, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* AF, lapack_int ldaf,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_zgerfs(
        LAPACK_COL_MAJOR, trans, n, nrhs,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) AF, ldaf,
        ipiv,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) X, ldx,
        ferr,
        berr );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gerqf(
    lapack_int m, lapack_int n,
    float* A, lapack_int lda,
    float* tau )
{
    return LAPACKE_sgerqf(
        LAPACK_COL_MAJOR, m, n,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_gerqf(
    lapack_int m, lapack_int n,
    double* A, lapack_int lda,
    double* tau )
{
    return LAPACKE_dgerqf(
        LAPACK_COL_MAJOR, m, n,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_gerqf(
    lapack_int m, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* tau )
{
    return LAPACKE_cgerqf(
        LAPACK_COL_MAJOR, m, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) tau );
}

inline lapack_int LAPACKE_gerqf(
    lapack_int m, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* tau )
{
    return LAPACKE_zgerqf(
        LAPACK_COL_MAJOR, m, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) tau );
}

//------------------------------------------------------------------------------
// Simple overloaded wrappers around LAPACKE (assuming routines in LAPACKE).
// These should go in test/lapacke_wrappers.hh.
inline lapack_int LAPACKE_gemqrt(
    char side, char trans, lapack_int m, lapack_int n, lapack_int k, lapack_int nb,
    float* V, lapack_int ldv,
    float* T, lapack_int ldt,
    float* C, lapack_int ldc )
{
    if (trans == 'C')
        trans = 'T';
    return LAPACKE_sgemqrt(
        LAPACK_COL_MAJOR, side, trans, m, n, k, nb,
        V, ldv,
        T, ldt,
        C, ldc );
}

inline lapack_int LAPACKE_gemqrt(
    char side, char trans, lapack_int m, lapack_int n, lapack_int k, lapack_int nb,
    double* V, lapack_int ldv,
    double* T, lapack_int ldt,
    double* C, lapack_int ldc )
{
    if (trans == 'C')
        trans = 'T';
    return LAPACKE_dgemqrt(
        LAPACK_COL_MAJOR, side, trans, m, n, k, nb,
        V, ldv,
        T, ldt,
        C, ldc );
}

inline lapack_int LAPACKE_gemqrt(
    char side, char trans, lapack_int m, lapack_int n, lapack_int k, lapack_int nb,
    std::complex<float>* V, lapack_int ldv,
    std::complex<float>* T, lapack_int ldt,
    std::complex<float>* C, lapack_int ldc )
{
    return LAPACKE_cgemqrt(
        LAPACK_COL_MAJOR, side, trans, m, n, k, nb,
        (lapack_complex_float*) V, ldv,
        (lapack_complex_float*) T, ldt,
        (lapack_complex_float*) C, ldc );
}

inline lapack_int LAPACKE_gemqrt(
    char side, char trans, lapack_int m, lapack_int n, lapack_int k, lapack_int nb,
    std::complex<double>* V, lapack_int ldv,
    std::complex<double>* T, lapack_int ldt,
    std::complex<double>* C, lapack_int ldc )
{
    return LAPACKE_zgemqrt(
        LAPACK_COL_MAJOR, side, trans, m, n, k, nb,
        (lapack_complex_double*) V, ldv,
        (lapack_complex_double*) T, ldt,
        (lapack_complex_double*) C, ldc );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gesdd(
    char jobz, lapack_int m, lapack_int n,
    float* A, lapack_int lda,
    float* S,
    float* U, lapack_int ldu,
    float* VT, lapack_int ldvt )
{
    return LAPACKE_sgesdd(
        LAPACK_COL_MAJOR, jobz, m, n,
        A, lda,
        S,
        U, ldu,
        VT, ldvt );
}

inline lapack_int LAPACKE_gesdd(
    char jobz, lapack_int m, lapack_int n,
    double* A, lapack_int lda,
    double* S,
    double* U, lapack_int ldu,
    double* VT, lapack_int ldvt )
{
    return LAPACKE_dgesdd(
        LAPACK_COL_MAJOR, jobz, m, n,
        A, lda,
        S,
        U, ldu,
        VT, ldvt );
}

inline lapack_int LAPACKE_gesdd(
    char jobz, lapack_int m, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    float* S,
    std::complex<float>* U, lapack_int ldu,
    std::complex<float>* VT, lapack_int ldvt )
{
    return LAPACKE_cgesdd(
        LAPACK_COL_MAJOR, jobz, m, n,
        (lapack_complex_float*) A, lda,
        S,
        (lapack_complex_float*) U, ldu,
        (lapack_complex_float*) VT, ldvt );
}

inline lapack_int LAPACKE_gesdd(
    char jobz, lapack_int m, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    double* S,
    std::complex<double>* U, lapack_int ldu,
    std::complex<double>* VT, lapack_int ldvt )
{
    return LAPACKE_zgesdd(
        LAPACK_COL_MAJOR, jobz, m, n,
        (lapack_complex_double*) A, lda,
        S,
        (lapack_complex_double*) U, ldu,
        (lapack_complex_double*) VT, ldvt );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gesv(
    lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_sgesv(
        LAPACK_COL_MAJOR, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_gesv(
    lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dgesv(
        LAPACK_COL_MAJOR, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_gesv(
    lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cgesv(
        LAPACK_COL_MAJOR, n, nrhs,
        (lapack_complex_float*) A, lda,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_gesv(
    lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zgesv(
        LAPACK_COL_MAJOR, n, nrhs,
        (lapack_complex_double*) A, lda,
        ipiv,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gesvd(
    char jobu, char jobvt, lapack_int m, lapack_int n,
    float* A, lapack_int lda,
    float* S,
    float* U, lapack_int ldu,
    float* VT, lapack_int ldvt )
{
    std::vector< float > superdiag( std::min( m, n ));
    return LAPACKE_sgesvd(
        LAPACK_COL_MAJOR, jobu, jobvt, m, n,
        A, lda,
        S,
        U, ldu,
        VT, ldvt,
        &superdiag[0] );
}

inline lapack_int LAPACKE_gesvd(
    char jobu, char jobvt, lapack_int m, lapack_int n,
    double* A, lapack_int lda,
    double* S,
    double* U, lapack_int ldu,
    double* VT, lapack_int ldvt )
{
    std::vector< double > superdiag( std::min( m, n ));
    return LAPACKE_dgesvd(
        LAPACK_COL_MAJOR, jobu, jobvt, m, n,
        A, lda,
        S,
        U, ldu,
        VT, ldvt,
        &superdiag[0] );
}

inline lapack_int LAPACKE_gesvd(
    char jobu, char jobvt, lapack_int m, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    float* S,
    std::complex<float>* U, lapack_int ldu,
    std::complex<float>* VT, lapack_int ldvt )
{
    std::vector< float > superdiag( std::min( m, n ));
    return LAPACKE_cgesvd(
        LAPACK_COL_MAJOR, jobu, jobvt, m, n,
        (lapack_complex_float*) A, lda,
        S,
        (lapack_complex_float*) U, ldu,
        (lapack_complex_float*) VT, ldvt,
        &superdiag[0] );
}

inline lapack_int LAPACKE_gesvd(
    char jobu, char jobvt, lapack_int m, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    double* S,
    std::complex<double>* U, lapack_int ldu,
    std::complex<double>* VT, lapack_int ldvt )
{
    std::vector< double > superdiag( std::min( m, n ));
    return LAPACKE_zgesvd(
        LAPACK_COL_MAJOR, jobu, jobvt, m, n,
        (lapack_complex_double*) A, lda,
        S,
        (lapack_complex_double*) U, ldu,
        (lapack_complex_double*) VT, ldvt,
        &superdiag[0] );
}

// -----------------------------------------------------------------------------
#if LAPACK_VERSION >= 30600
inline lapack_int LAPACKE_gesvdx(
    char jobu, char jobvt, char range, lapack_int m, lapack_int n,
    float* A, lapack_int lda,
    float vl, float vu, lapack_int il, lapack_int iu, lapack_int* nfound,
    float* S,
    float* U, lapack_int ldu,
    float* VT, lapack_int ldvt )
{
    int64_t minmn = ( m < n ? m : n );
    std::vector< lapack_int > superb( 12 * minmn );
    return LAPACKE_sgesvdx(
        LAPACK_COL_MAJOR, jobu, jobvt, range, m, n,
        A, lda, vl, vu, il, iu,
        nfound,
        S,
        U, ldu,
        VT, ldvt,
        &superb[0] );
}

inline lapack_int LAPACKE_gesvdx(
    char jobu, char jobvt, char range, lapack_int m, lapack_int n,
    double* A, lapack_int lda,
    double vl, double vu, lapack_int il, lapack_int iu, lapack_int* nfound,
    double* S,
    double* U, lapack_int ldu,
    double* VT, lapack_int ldvt )
{
    int64_t minmn = ( m < n ? m : n );
    std::vector< lapack_int > superb( 12 * minmn );
    return LAPACKE_dgesvdx(
        LAPACK_COL_MAJOR, jobu, jobvt, range, m, n,
        A, lda, vl, vu, il, iu,
        nfound,
        S,
        U, ldu,
        VT, ldvt,
        &superb[0] );
}

inline lapack_int LAPACKE_gesvdx(
    char jobu, char jobvt, char range, lapack_int m, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    float vl, float vu, lapack_int il, lapack_int iu, lapack_int* nfound,
    float* S,
    std::complex<float>* U, lapack_int ldu,
    std::complex<float>* VT, lapack_int ldvt )
{
    int64_t minmn = ( m < n ? m : n );
    std::vector< lapack_int > superb( 12 * minmn );
    return LAPACKE_cgesvdx(
        LAPACK_COL_MAJOR, jobu, jobvt, range, m, n,
        (lapack_complex_float*) A, lda, vl, vu, il, iu,
        nfound,
        S,
        (lapack_complex_float*) U, ldu,
        (lapack_complex_float*) VT, ldvt,
        &superb[0] );
}

inline lapack_int LAPACKE_gesvdx(
    char jobu, char jobvt, char range, lapack_int m, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    double vl, double vu, lapack_int il, lapack_int iu, lapack_int* nfound,
    double* S,
    std::complex<double>* U, lapack_int ldu,
    std::complex<double>* VT, lapack_int ldvt )
{
    int64_t minmn = ( m < n ? m : n );
    std::vector< lapack_int > superb( 12 * minmn );
    return LAPACKE_zgesvdx(
        LAPACK_COL_MAJOR, jobu, jobvt, range, m, n,
        (lapack_complex_double*) A, lda, vl, vu, il, iu,
        nfound,
        S,
        (lapack_complex_double*) U, ldu,
        (lapack_complex_double*) VT, ldvt,
        &superb[0] );
}
#endif // 30600

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gesvx(
    char fact, char trans, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    float* AF, lapack_int ldaf,
    lapack_int* ipiv, char* equed,
    float* R, float* C,
    float* B, lapack_int ldb,
    float* X, lapack_int ldx,
    float* rcond,
    float* ferr, float* berr,
    float* rpivot )
{
    return LAPACKE_sgesvx(
        LAPACK_COL_MAJOR, fact, trans, n, nrhs,
        A, lda,
        AF, ldaf,
        ipiv,
        equed,
        R, C,
        B, ldb,
        X, ldx,
        rcond,
        ferr, berr,
        rpivot );
}

inline lapack_int LAPACKE_gesvx(
    char fact, char trans, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    double* AF, lapack_int ldaf,
    lapack_int* ipiv, char* equed,
    double* R, double* C,
    double* B, lapack_int ldb,
    double* X, lapack_int ldx,
    double* rcond,
    double* ferr, double* berr,
    double* rpivot )
{
    return LAPACKE_dgesvx(
        LAPACK_COL_MAJOR, fact, trans, n, nrhs,
        A, lda,
        AF, ldaf,
        ipiv,
        equed,
        R, C,
        B, ldb,
        X, ldx,
        rcond,
        ferr, berr,
        rpivot );
}

inline lapack_int LAPACKE_gesvx(
    char fact, char trans, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* AF, lapack_int ldaf,
    lapack_int* ipiv, char* equed,
    float* R, float* C,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* X, lapack_int ldx,
    float* rcond,
    float* ferr, float* berr,
    float* rpivot )
{
    return LAPACKE_cgesvx(
        LAPACK_COL_MAJOR, fact, trans, n, nrhs,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) AF, ldaf,
        ipiv,
        equed,
        R, C,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) X, ldx,
        rcond,
        ferr, berr,
        rpivot );
}

inline lapack_int LAPACKE_gesvx(
    char fact, char trans, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* AF, lapack_int ldaf,
    lapack_int* ipiv, char* equed,
    double* R,
    double* C,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* X, lapack_int ldx,
    double* rcond,
    double* ferr, double* berr,
    double* rpivot )
{
    return LAPACKE_zgesvx(
        LAPACK_COL_MAJOR, fact, trans, n, nrhs,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) AF, ldaf,
        ipiv,
        equed,
        R,
        C,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) X, ldx,
        rcond,
        ferr, berr,
        rpivot );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_getrf(
    lapack_int m, lapack_int n,
    float* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_sgetrf(
        LAPACK_COL_MAJOR, m, n,
        A, lda,
        ipiv );
}

inline lapack_int LAPACKE_getrf(
    lapack_int m, lapack_int n,
    double* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_dgetrf(
        LAPACK_COL_MAJOR, m, n,
        A, lda,
        ipiv );
}

inline lapack_int LAPACKE_getrf(
    lapack_int m, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_cgetrf(
        LAPACK_COL_MAJOR, m, n,
        (lapack_complex_float*) A, lda,
        ipiv );
}

inline lapack_int LAPACKE_getrf(
    lapack_int m, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_zgetrf(
        LAPACK_COL_MAJOR, m, n,
        (lapack_complex_double*) A, lda,
        ipiv );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_getri(
    lapack_int n,
    float* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_sgetri(
        LAPACK_COL_MAJOR, n,
        A, lda,
        ipiv );
}

inline lapack_int LAPACKE_getri(
    lapack_int n,
    double* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_dgetri(
        LAPACK_COL_MAJOR, n,
        A, lda,
        ipiv );
}

inline lapack_int LAPACKE_getri(
    lapack_int n,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_cgetri(
        LAPACK_COL_MAJOR, n,
        (lapack_complex_float*) A, lda,
        ipiv );
}

inline lapack_int LAPACKE_getri(
    lapack_int n,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_zgetri(
        LAPACK_COL_MAJOR, n,
        (lapack_complex_double*) A, lda,
        ipiv );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_getrs(
    char trans, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_sgetrs(
        LAPACK_COL_MAJOR, trans, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_getrs(
    char trans, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dgetrs(
        LAPACK_COL_MAJOR, trans, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_getrs(
    char trans, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cgetrs(
        LAPACK_COL_MAJOR, trans, n, nrhs,
        (lapack_complex_float*) A, lda,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_getrs(
    char trans, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zgetrs(
        LAPACK_COL_MAJOR, trans, n, nrhs,
        (lapack_complex_double*) A, lda,
        ipiv,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
#if LAPACK_VERSION >= 30700
inline lapack_int LAPACKE_getsls(
    char trans, lapack_int m, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    float* B, lapack_int ldb )
{
    if (trans == 'C')
        trans = 'T';
    return LAPACKE_sgetsls(
        LAPACK_COL_MAJOR, trans, m, n, nrhs,
        A, lda,
        B, ldb );
}

inline lapack_int LAPACKE_getsls(
    char trans, lapack_int m, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    double* B, lapack_int ldb )
{
    if (trans == 'C')
        trans = 'T';
    return LAPACKE_dgetsls(
        LAPACK_COL_MAJOR, trans, m, n, nrhs,
        A, lda,
        B, ldb );
}

inline lapack_int LAPACKE_getsls(
    char trans, lapack_int m, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb )
{
    if (trans == 'T')
        trans = 'C';
    return LAPACKE_cgetsls(
        LAPACK_COL_MAJOR, trans, m, n, nrhs,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_getsls(
    char trans, lapack_int m, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb )
{
    if (trans == 'T')
        trans = 'C';
    return LAPACKE_zgetsls(
        LAPACK_COL_MAJOR, trans, m, n, nrhs,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb );
}
#endif // 30700

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_ggev(
    char jobvl, char jobvr, lapack_int n,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    std::complex<float>* alpha,
    float* beta,
    float* VL, lapack_int ldvl,
    float* VR, lapack_int ldvr )
{
    std::vector< float > alphar( n ), alphai( n );
    lapack_int err = LAPACKE_sggev(
        LAPACK_COL_MAJOR, jobvl, jobvr, n,
        A, lda,
        B, ldb,
        &alphar[0], &alphai[0],
        beta,
        VL, ldvl,
        VR, ldvr );
    for (int64_t i = 0; i < n; ++i) {
        alpha[i] = std::complex<float>( alphar[i], alphai[i] );
    }
    return err;
}

inline lapack_int LAPACKE_ggev(
    char jobvl, char jobvr, lapack_int n,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    std::complex<double>* alpha,
    double* beta,
    double* VL, lapack_int ldvl,
    double* VR, lapack_int ldvr )
{
    std::vector< double > alphar( n ), alphai( n );
    lapack_int err = LAPACKE_dggev(
        LAPACK_COL_MAJOR, jobvl, jobvr, n,
        A, lda,
        B, ldb,
        &alphar[0], &alphai[0],
        beta,
        VL, ldvl,
        VR, ldvr );
    for (int64_t i = 0; i < n; ++i) {
        alpha[i] = std::complex<double>( alphar[i], alphai[i] );
    }
    return err;
}

inline lapack_int LAPACKE_ggev(
    char jobvl, char jobvr, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* alpha,
    std::complex<float>* beta,
    std::complex<float>* VL, lapack_int ldvl,
    std::complex<float>* VR, lapack_int ldvr )
{
    return LAPACKE_cggev(
        LAPACK_COL_MAJOR, jobvl, jobvr, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) alpha,
        (lapack_complex_float*) beta,
        (lapack_complex_float*) VL, ldvl,
        (lapack_complex_float*) VR, ldvr );
}

inline lapack_int LAPACKE_ggev(
    char jobvl, char jobvr, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* alpha,
    std::complex<double>* beta,
    std::complex<double>* VL, lapack_int ldvl,
    std::complex<double>* VR, lapack_int ldvr )
{
    return LAPACKE_zggev(
        LAPACK_COL_MAJOR, jobvl, jobvr, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) alpha,
        (lapack_complex_double*) beta,
        (lapack_complex_double*) VL, ldvl,
        (lapack_complex_double*) VR, ldvr );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_ggglm(
    lapack_int n, lapack_int m, lapack_int p,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    float* D,
    float* X,
    float* Y )
{
    return LAPACKE_sggglm(
        LAPACK_COL_MAJOR, n, m, p,
        A, lda,
        B, ldb,
        D,
        X,
        Y );
}

inline lapack_int LAPACKE_ggglm(
    lapack_int n, lapack_int m, lapack_int p,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    double* D,
    double* X,
    double* Y )
{
    return LAPACKE_dggglm(
        LAPACK_COL_MAJOR, n, m, p,
        A, lda,
        B, ldb,
        D,
        X,
        Y );
}

inline lapack_int LAPACKE_ggglm(
    lapack_int n, lapack_int m, lapack_int p,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* D,
    std::complex<float>* X,
    std::complex<float>* Y )
{
    return LAPACKE_cggglm(
        LAPACK_COL_MAJOR, n, m, p,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) D,
        (lapack_complex_float*) X,
        (lapack_complex_float*) Y );
}

inline lapack_int LAPACKE_ggglm(
    lapack_int n, lapack_int m, lapack_int p,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* D,
    std::complex<double>* X,
    std::complex<double>* Y )
{
    return LAPACKE_zggglm(
        LAPACK_COL_MAJOR, n, m, p,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) D,
        (lapack_complex_double*) X,
        (lapack_complex_double*) Y );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gglse(
    lapack_int m, lapack_int n, lapack_int p,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    float* C,
    float* D,
    float* X )
{
    return LAPACKE_sgglse(
        LAPACK_COL_MAJOR, m, n, p,
        A, lda,
        B, ldb,
        C,
        D,
        X );
}

inline lapack_int LAPACKE_gglse(
    lapack_int m, lapack_int n, lapack_int p,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    double* C,
    double* D,
    double* X )
{
    return LAPACKE_dgglse(
        LAPACK_COL_MAJOR, m, n, p,
        A, lda,
        B, ldb,
        C,
        D,
        X );
}

inline lapack_int LAPACKE_gglse(
    lapack_int m, lapack_int n, lapack_int p,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* C,
    std::complex<float>* D,
    std::complex<float>* X )
{
    return LAPACKE_cgglse(
        LAPACK_COL_MAJOR, m, n, p,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) C,
        (lapack_complex_float*) D,
        (lapack_complex_float*) X );
}

inline lapack_int LAPACKE_gglse(
    lapack_int m, lapack_int n, lapack_int p,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* C,
    std::complex<double>* D,
    std::complex<double>* X )
{
    return LAPACKE_zgglse(
        LAPACK_COL_MAJOR, m, n, p,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) C,
        (lapack_complex_double*) D,
        (lapack_complex_double*) X );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_ggqrf(
    lapack_int n, lapack_int m, lapack_int p,
    float* A, lapack_int lda,
    float* taua,
    float* B, lapack_int ldb,
    float* taub )
{
    return LAPACKE_sggqrf(
        LAPACK_COL_MAJOR, n, m, p,
        A, lda,
        taua,
        B, ldb,
        taub );
}

inline lapack_int LAPACKE_ggqrf(
    lapack_int n, lapack_int m, lapack_int p,
    double* A, lapack_int lda,
    double* taua,
    double* B, lapack_int ldb,
    double* taub )
{
    return LAPACKE_dggqrf(
        LAPACK_COL_MAJOR, n, m, p,
        A, lda,
        taua,
        B, ldb,
        taub );
}

inline lapack_int LAPACKE_ggqrf(
    lapack_int n, lapack_int m, lapack_int p,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* taua,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* taub )
{
    return LAPACKE_cggqrf(
        LAPACK_COL_MAJOR, n, m, p,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) taua,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) taub );
}

inline lapack_int LAPACKE_ggqrf(
    lapack_int n, lapack_int m, lapack_int p,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* taua,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* taub )
{
    return LAPACKE_zggqrf(
        LAPACK_COL_MAJOR, n, m, p,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) taua,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) taub );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_ggrqf(
    lapack_int m, lapack_int p, lapack_int n,
    float* A, lapack_int lda,
    float* taua,
    float* B, lapack_int ldb,
    float* taub )
{
    return LAPACKE_sggrqf(
        LAPACK_COL_MAJOR, m, p, n,
        A, lda,
        taua,
        B, ldb,
        taub );
}

inline lapack_int LAPACKE_ggrqf(
    lapack_int m, lapack_int p, lapack_int n,
    double* A, lapack_int lda,
    double* taua,
    double* B, lapack_int ldb,
    double* taub )
{
    return LAPACKE_dggrqf(
        LAPACK_COL_MAJOR, m, p, n,
        A, lda,
        taua,
        B, ldb,
        taub );
}

inline lapack_int LAPACKE_ggrqf(
    lapack_int m, lapack_int p, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* taua,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* taub )
{
    return LAPACKE_cggrqf(
        LAPACK_COL_MAJOR, m, p, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) taua,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) taub );
}

inline lapack_int LAPACKE_ggrqf(
    lapack_int m, lapack_int p, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* taua,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* taub )
{
    return LAPACKE_zggrqf(
        LAPACK_COL_MAJOR, m, p, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) taua,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) taub );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gtcon(
    char norm, lapack_int n,
    float* DL,
    float* D,
    float* DU,
    float* DU2,
    lapack_int* ipiv, float anorm,
    float* rcond )
{
    return LAPACKE_sgtcon(
        norm, n,
        DL,
        D,
        DU,
        DU2,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_gtcon(
    char norm, lapack_int n,
    double* DL,
    double* D,
    double* DU,
    double* DU2,
    lapack_int* ipiv, double anorm,
    double* rcond )
{
    return LAPACKE_dgtcon(
        norm, n,
        DL,
        D,
        DU,
        DU2,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_gtcon(
    char norm, lapack_int n,
    std::complex<float>* DL,
    std::complex<float>* D,
    std::complex<float>* DU,
    std::complex<float>* DU2,
    lapack_int* ipiv, float anorm,
    float* rcond )
{
    return LAPACKE_cgtcon(
        norm, n,
        (lapack_complex_float*) DL,
        (lapack_complex_float*) D,
        (lapack_complex_float*) DU,
        (lapack_complex_float*) DU2,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_gtcon(
    char norm, lapack_int n,
    std::complex<double>* DL,
    std::complex<double>* D,
    std::complex<double>* DU,
    std::complex<double>* DU2,
    lapack_int* ipiv, double anorm,
    double* rcond )
{
    return LAPACKE_zgtcon(
        norm, n,
        (lapack_complex_double*) DL,
        (lapack_complex_double*) D,
        (lapack_complex_double*) DU,
        (lapack_complex_double*) DU2,
        ipiv, anorm,
        rcond );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gtrfs(
    char trans, lapack_int n, lapack_int nrhs,
    float* DL,
    float* D,
    float* DU,
    float* DLF,
    float* DF,
    float* DUF,
    float* DU2,
    lapack_int* ipiv,
    float* B, lapack_int ldb,
    float* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_sgtrfs(
        LAPACK_COL_MAJOR, trans, n, nrhs,
        DL,
        D,
        DU,
        DLF,
        DF,
        DUF,
        DU2,
        ipiv,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_gtrfs(
    char trans, lapack_int n, lapack_int nrhs,
    double* DL,
    double* D,
    double* DU,
    double* DLF,
    double* DF,
    double* DUF,
    double* DU2,
    lapack_int* ipiv,
    double* B, lapack_int ldb,
    double* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_dgtrfs(
        LAPACK_COL_MAJOR, trans, n, nrhs,
        DL,
        D,
        DU,
        DLF,
        DF,
        DUF,
        DU2,
        ipiv,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_gtrfs(
    char trans, lapack_int n, lapack_int nrhs,
    std::complex<float>* DL,
    std::complex<float>* D,
    std::complex<float>* DU,
    std::complex<float>* DLF,
    std::complex<float>* DF,
    std::complex<float>* DUF,
    std::complex<float>* DU2,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_cgtrfs(
        LAPACK_COL_MAJOR, trans, n, nrhs,
        (lapack_complex_float*) DL,
        (lapack_complex_float*) D,
        (lapack_complex_float*) DU,
        (lapack_complex_float*) DLF,
        (lapack_complex_float*) DF,
        (lapack_complex_float*) DUF,
        (lapack_complex_float*) DU2,
        ipiv,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_gtrfs(
    char trans, lapack_int n, lapack_int nrhs,
    std::complex<double>* DL,
    std::complex<double>* D,
    std::complex<double>* DU,
    std::complex<double>* DLF,
    std::complex<double>* DF,
    std::complex<double>* DUF,
    std::complex<double>* DU2,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_zgtrfs(
        LAPACK_COL_MAJOR, trans, n, nrhs,
        (lapack_complex_double*) DL,
        (lapack_complex_double*) D,
        (lapack_complex_double*) DU,
        (lapack_complex_double*) DLF,
        (lapack_complex_double*) DF,
        (lapack_complex_double*) DUF,
        (lapack_complex_double*) DU2,
        ipiv,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) X, ldx,
        ferr,
        berr );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gtsv(
    lapack_int n, lapack_int nrhs,
    float* DL,
    float* D,
    float* DU,
    float* B, lapack_int ldb )
{
    return LAPACKE_sgtsv(
        LAPACK_COL_MAJOR, n, nrhs,
        DL,
        D,
        DU,
        B, ldb );
}

inline lapack_int LAPACKE_gtsv(
    lapack_int n, lapack_int nrhs,
    double* DL,
    double* D,
    double* DU,
    double* B, lapack_int ldb )
{
    return LAPACKE_dgtsv(
        LAPACK_COL_MAJOR, n, nrhs,
        DL,
        D,
        DU,
        B, ldb );
}

inline lapack_int LAPACKE_gtsv(
    lapack_int n, lapack_int nrhs,
    std::complex<float>* DL,
    std::complex<float>* D,
    std::complex<float>* DU,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cgtsv(
        LAPACK_COL_MAJOR, n, nrhs,
        (lapack_complex_float*) DL,
        (lapack_complex_float*) D,
        (lapack_complex_float*) DU,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_gtsv(
    lapack_int n, lapack_int nrhs,
    std::complex<double>* DL,
    std::complex<double>* D,
    std::complex<double>* DU,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zgtsv(
        LAPACK_COL_MAJOR, n, nrhs,
        (lapack_complex_double*) DL,
        (lapack_complex_double*) D,
        (lapack_complex_double*) DU,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gttrf(
    lapack_int n,
    float* DL,
    float* D,
    float* DU,
    float* DU2,
    lapack_int* ipiv )
{
    return LAPACKE_sgttrf(
        n,
        DL,
        D,
        DU,
        DU2,
        ipiv );
}

inline lapack_int LAPACKE_gttrf(
    lapack_int n,
    double* DL,
    double* D,
    double* DU,
    double* DU2,
    lapack_int* ipiv )
{
    return LAPACKE_dgttrf(
        n,
        DL,
        D,
        DU,
        DU2,
        ipiv );
}

inline lapack_int LAPACKE_gttrf(
    lapack_int n,
    std::complex<float>* DL,
    std::complex<float>* D,
    std::complex<float>* DU,
    std::complex<float>* DU2,
    lapack_int* ipiv )
{
    return LAPACKE_cgttrf(
        n,
        (lapack_complex_float*) DL,
        (lapack_complex_float*) D,
        (lapack_complex_float*) DU,
        (lapack_complex_float*) DU2,
        ipiv );
}

inline lapack_int LAPACKE_gttrf(
    lapack_int n,
    std::complex<double>* DL,
    std::complex<double>* D,
    std::complex<double>* DU,
    std::complex<double>* DU2,
    lapack_int* ipiv )
{
    return LAPACKE_zgttrf(
        n,
        (lapack_complex_double*) DL,
        (lapack_complex_double*) D,
        (lapack_complex_double*) DU,
        (lapack_complex_double*) DU2,
        ipiv );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_gttrs(
    char trans, lapack_int n, lapack_int nrhs,
    float* DL,
    float* D,
    float* DU,
    float* DU2,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_sgttrs(
        LAPACK_COL_MAJOR, trans, n, nrhs,
        DL,
        D,
        DU,
        DU2,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_gttrs(
    char trans, lapack_int n, lapack_int nrhs,
    double* DL,
    double* D,
    double* DU,
    double* DU2,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dgttrs(
        LAPACK_COL_MAJOR, trans, n, nrhs,
        DL,
        D,
        DU,
        DU2,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_gttrs(
    char trans, lapack_int n, lapack_int nrhs,
    std::complex<float>* DL,
    std::complex<float>* D,
    std::complex<float>* DU,
    std::complex<float>* DU2,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cgttrs(
        LAPACK_COL_MAJOR, trans, n, nrhs,
        (lapack_complex_float*) DL,
        (lapack_complex_float*) D,
        (lapack_complex_float*) DU,
        (lapack_complex_float*) DU2,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_gttrs(
    char trans, lapack_int n, lapack_int nrhs,
    std::complex<double>* DL,
    std::complex<double>* D,
    std::complex<double>* DU,
    std::complex<double>* DU2,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zgttrs(
        LAPACK_COL_MAJOR, trans, n, nrhs,
        (lapack_complex_double*) DL,
        (lapack_complex_double*) D,
        (lapack_complex_double*) DU,
        (lapack_complex_double*) DU2,
        ipiv,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hbev(
    char jobz, char uplo, lapack_int n, lapack_int kd,
    float* AB, lapack_int ldab,
    float* W,
    float* Z, lapack_int ldz )
{
    return LAPACKE_ssbev(
        LAPACK_COL_MAJOR, jobz, uplo, n, kd,
        AB, ldab,
        W,
        Z, ldz );
}

inline lapack_int LAPACKE_hbev(
    char jobz, char uplo, lapack_int n, lapack_int kd,
    double* AB, lapack_int ldab,
    double* W,
    double* Z, lapack_int ldz )
{
    return LAPACKE_dsbev(
        LAPACK_COL_MAJOR, jobz, uplo, n, kd,
        AB, ldab,
        W,
        Z, ldz );
}

inline lapack_int LAPACKE_hbev(
    char jobz, char uplo, lapack_int n, lapack_int kd,
    std::complex<float>* AB, lapack_int ldab,
    float* W,
    std::complex<float>* Z, lapack_int ldz )
{
    return LAPACKE_chbev(
        LAPACK_COL_MAJOR, jobz, uplo, n, kd,
        (lapack_complex_float*) AB, ldab,
        W,
        (lapack_complex_float*) Z, ldz );
}

inline lapack_int LAPACKE_hbev(
    char jobz, char uplo, lapack_int n, lapack_int kd,
    std::complex<double>* AB, lapack_int ldab,
    double* W,
    std::complex<double>* Z, lapack_int ldz )
{
    return LAPACKE_zhbev(
        LAPACK_COL_MAJOR, jobz, uplo, n, kd,
        (lapack_complex_double*) AB, ldab,
        W,
        (lapack_complex_double*) Z, ldz );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hbevd(
    char jobz, char uplo, lapack_int n, lapack_int kd,
    float* AB, lapack_int ldab,
    float* W,
    float* Z, lapack_int ldz )
{
    return LAPACKE_ssbevd(
        LAPACK_COL_MAJOR, jobz, uplo, n, kd,
        AB, ldab,
        W,
        Z, ldz );
}

inline lapack_int LAPACKE_hbevd(
    char jobz, char uplo, lapack_int n, lapack_int kd,
    double* AB, lapack_int ldab,
    double* W,
    double* Z, lapack_int ldz )
{
    return LAPACKE_dsbevd(
        LAPACK_COL_MAJOR, jobz, uplo, n, kd,
        AB, ldab,
        W,
        Z, ldz );
}

inline lapack_int LAPACKE_hbevd(
    char jobz, char uplo, lapack_int n, lapack_int kd,
    std::complex<float>* AB, lapack_int ldab,
    float* W,
    std::complex<float>* Z, lapack_int ldz )
{
    return LAPACKE_chbevd(
        LAPACK_COL_MAJOR, jobz, uplo, n, kd,
        (lapack_complex_float*) AB, ldab,
        W,
        (lapack_complex_float*) Z, ldz );
}

inline lapack_int LAPACKE_hbevd(
    char jobz, char uplo, lapack_int n, lapack_int kd,
    std::complex<double>* AB, lapack_int ldab,
    double* W,
    std::complex<double>* Z, lapack_int ldz )
{
    return LAPACKE_zhbevd(
        LAPACK_COL_MAJOR, jobz, uplo, n, kd,
        (lapack_complex_double*) AB, ldab,
        W,
        (lapack_complex_double*) Z, ldz );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hbevx(
    char jobz, char range, char uplo, lapack_int n, lapack_int kd,
    float* AB, lapack_int ldab,
    float* Q, lapack_int ldq,
    float vl, float vu, lapack_int il, lapack_int iu, float abstol,
    lapack_int* nfound, float* W,
    float* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_ssbevx(
        LAPACK_COL_MAJOR, jobz, range, uplo, n, kd,
        AB, ldab,
        Q, ldq, vl, vu, il, iu, abstol,
        nfound, W,
        Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_hbevx(
    char jobz, char range, char uplo, lapack_int n, lapack_int kd,
    double* AB, lapack_int ldab,
    double* Q, lapack_int ldq,
    double vl, double vu, lapack_int il, lapack_int iu, double abstol,
    lapack_int* nfound, double* W,
    double* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_dsbevx(
        LAPACK_COL_MAJOR, jobz, range, uplo, n, kd,
        AB, ldab,
        Q, ldq, vl, vu, il, iu, abstol,
        nfound, W,
        Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_hbevx(
    char jobz, char range, char uplo, lapack_int n, lapack_int kd,
    std::complex<float>* AB, lapack_int ldab,
    std::complex<float>* Q, lapack_int ldq,
    float vl, float vu, lapack_int il, lapack_int iu, float abstol,
    lapack_int* nfound, float* W,
    std::complex<float>* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_chbevx(
        LAPACK_COL_MAJOR, jobz, range, uplo, n, kd,
        (lapack_complex_float*) AB, ldab,
        (lapack_complex_float*) Q, ldq, vl, vu, il, iu, abstol,
        nfound, W,
        (lapack_complex_float*) Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_hbevx(
    char jobz, char range, char uplo, lapack_int n, lapack_int kd,
    std::complex<double>* AB, lapack_int ldab,
    std::complex<double>* Q, lapack_int ldq,
    double vl, double vu, lapack_int il, lapack_int iu, double abstol,
    lapack_int* nfound, double* W,
    std::complex<double>* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_zhbevx(
        LAPACK_COL_MAJOR, jobz, range, uplo, n, kd,
        (lapack_complex_double*) AB, ldab,
        (lapack_complex_double*) Q, ldq, vl, vu, il, iu, abstol,
        nfound, W,
        (lapack_complex_double*) Z, ldz,
        ifail );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hbgv(
    char jobz, char uplo, lapack_int n, lapack_int ka, lapack_int kb,
    float* AB, lapack_int ldab,
    float* BB, lapack_int ldbb,
    float* W,
    float* Z, lapack_int ldz )
{
    return LAPACKE_ssbgv(
        LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb,
        AB, ldab,
        BB, ldbb,
        W,
        Z, ldz );
}

inline lapack_int LAPACKE_hbgv(
    char jobz, char uplo, lapack_int n, lapack_int ka, lapack_int kb,
    double* AB, lapack_int ldab,
    double* BB, lapack_int ldbb,
    double* W,
    double* Z, lapack_int ldz )
{
    return LAPACKE_dsbgv(
        LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb,
        AB, ldab,
        BB, ldbb,
        W,
        Z, ldz );
}

inline lapack_int LAPACKE_hbgv(
    char jobz, char uplo, lapack_int n, lapack_int ka, lapack_int kb,
    std::complex<float>* AB, lapack_int ldab,
    std::complex<float>* BB, lapack_int ldbb,
    float* W,
    std::complex<float>* Z, lapack_int ldz )
{
    return LAPACKE_chbgv(
        LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb,
        (lapack_complex_float*) AB, ldab,
        (lapack_complex_float*) BB, ldbb,
        W,
        (lapack_complex_float*) Z, ldz );
}

inline lapack_int LAPACKE_hbgv(
    char jobz, char uplo, lapack_int n, lapack_int ka, lapack_int kb,
    std::complex<double>* AB, lapack_int ldab,
    std::complex<double>* BB, lapack_int ldbb,
    double* W,
    std::complex<double>* Z, lapack_int ldz )
{
    return LAPACKE_zhbgv(
        LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb,
        (lapack_complex_double*) AB, ldab,
        (lapack_complex_double*) BB, ldbb,
        W,
        (lapack_complex_double*) Z, ldz );
}

// -----------------------------------------------------------------------------
// see custom version below
inline lapack_int LAPACKE_hbgvd(
    char jobz, char uplo, lapack_int n, lapack_int ka, lapack_int kb,
    float* AB, lapack_int ldab,
    float* BB, lapack_int ldbb,
    float* W,
    float* Z, lapack_int ldz )
{
    return LAPACKE_ssbgvd(
        LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb,
        AB, ldab,
        BB, ldbb,
        W,
        Z, ldz );
}

inline lapack_int LAPACKE_hbgvd(
    char jobz, char uplo, lapack_int n, lapack_int ka, lapack_int kb,
    double* AB, lapack_int ldab,
    double* BB, lapack_int ldbb,
    double* W,
    double* Z, lapack_int ldz )
{
    return LAPACKE_dsbgvd(
        LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb,
        AB, ldab,
        BB, ldbb,
        W,
        Z, ldz );
}

inline lapack_int LAPACKE_hbgvd(
    char jobz, char uplo, lapack_int n, lapack_int ka, lapack_int kb,
    std::complex<float>* AB, lapack_int ldab,
    std::complex<float>* BB, lapack_int ldbb,
    float* W,
    std::complex<float>* Z, lapack_int ldz )
{
    return LAPACKE_chbgvd(
        LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb,
        (lapack_complex_float*) AB, ldab,
        (lapack_complex_float*) BB, ldbb,
        W,
        (lapack_complex_float*) Z, ldz );
}

inline lapack_int LAPACKE_hbgvd(
    char jobz, char uplo, lapack_int n, lapack_int ka, lapack_int kb,
    std::complex<double>* AB, lapack_int ldab,
    std::complex<double>* BB, lapack_int ldbb,
    double* W,
    std::complex<double>* Z, lapack_int ldz )
{
    return LAPACKE_zhbgvd(
        LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb,
        (lapack_complex_double*) AB, ldab,
        (lapack_complex_double*) BB, ldbb,
        W,
        (lapack_complex_double*) Z, ldz );
}

// -----------------------------------------------------------------------------
// Note: LAPACKE_hbgvd does workspace query that may be wrong
// (e.g., in LAPACK <= 3.6.0, MKL 2018), so custom version fixes it.
inline lapack_int LAPACKE_hbgvd_custom(
    char jobz, char uplo, lapack_int n, lapack_int ka, lapack_int kb,
    float* AB, lapack_int ldab,
    float* BB, lapack_int ldbb,
    float* W,
    float* Z, lapack_int ldz )
{
    float query;
    lapack_int liwork;
    lapack_int info = LAPACKE_ssbgvd_work(
        LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb,
        AB, ldab,
        BB, ldbb,
        W,
        Z, ldz,
        &query, -1, &liwork, -1 );
    if (info < 0)
        return info;
    lapack_int lwork = query;
    // override potentially wrong query for LAPACK <= 3.6.0
    if (lwork < 3*n)
        lwork = 3*n;
    std::vector< float > work( lwork );
    std::vector< lapack_int > iwork( liwork );
    return LAPACKE_ssbgvd_work(
        LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb,
        AB, ldab,
        BB, ldbb,
        W,
        Z, ldz,
        &work[0], lwork, &iwork[0], liwork );
}

inline lapack_int LAPACKE_hbgvd_custom(
    char jobz, char uplo, lapack_int n, lapack_int ka, lapack_int kb,
    double* AB, lapack_int ldab,
    double* BB, lapack_int ldbb,
    double* W,
    double* Z, lapack_int ldz )
{
    double query;
    lapack_int liwork;
    lapack_int info = LAPACKE_dsbgvd_work(
        LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb,
        AB, ldab,
        BB, ldbb,
        W,
        Z, ldz,
        &query, -1, &liwork, -1 );
    if (info < 0)
        return info;
    lapack_int lwork = query;
    // override potentially wrong query for LAPACK <= 3.6.0
    if (lwork < 3*n)
        lwork = 3*n;
    std::vector< double > work( lwork );
    std::vector< lapack_int > iwork( liwork );
    return LAPACKE_dsbgvd_work(
        LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb,
        AB, ldab,
        BB, ldbb,
        W,
        Z, ldz,
        &work[0], lwork, &iwork[0], liwork );
}

inline lapack_int LAPACKE_hbgvd_custom(
    char jobz, char uplo, lapack_int n, lapack_int ka, lapack_int kb,
    std::complex<float>* AB, lapack_int ldab,
    std::complex<float>* BB, lapack_int ldbb,
    float* W,
    std::complex<float>* Z, lapack_int ldz )
{
    std::complex<float> query;
    float rquery;
    lapack_int liwork;
    lapack_int info = LAPACKE_chbgvd_work(
        LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb,
        (lapack_complex_float*) AB, ldab,
        (lapack_complex_float*) BB, ldbb,
        W,
        (lapack_complex_float*) Z, ldz,
        (lapack_complex_float*) &query, -1, &rquery, -1, &liwork, -1 );
    if (info < 0)
        return info;
    lapack_int lwork = real(query);
    lapack_int lrwork = rquery;
    // override potentially wrong query for LAPACK <= 3.6.0
    if (lrwork < 2*n)
        lrwork = 2*n;
    std::vector< std::complex<float> > work( lwork );
    std::vector< float > rwork( lrwork );
    std::vector< lapack_int > iwork( liwork );
    return LAPACKE_chbgvd_work(
        LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb,
        (lapack_complex_float*) AB, ldab,
        (lapack_complex_float*) BB, ldbb,
        W,
        (lapack_complex_float*) Z, ldz,
        (lapack_complex_float*) &work[0], lwork,
        &rwork[0], lrwork, &iwork[0], liwork );
}

inline lapack_int LAPACKE_hbgvd_custom(
    char jobz, char uplo, lapack_int n, lapack_int ka, lapack_int kb,
    std::complex<double>* AB, lapack_int ldab,
    std::complex<double>* BB, lapack_int ldbb,
    double* W,
    std::complex<double>* Z, lapack_int ldz )
{
    std::complex<double> query;
    double rquery;
    lapack_int liwork;
    lapack_int info = LAPACKE_zhbgvd_work(
        LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb,
        (lapack_complex_double*) AB, ldab,
        (lapack_complex_double*) BB, ldbb,
        W,
        (lapack_complex_double*) Z, ldz,
        (lapack_complex_double*) &query, -1, &rquery, -1, &liwork, -1 );
    if (info < 0)
        return info;
    lapack_int lwork = real(query);
    lapack_int lrwork = rquery;
    // override potentially wrong query for LAPACK <= 3.6.0
    if (lrwork < 2*n)
        lrwork = 2*n;
    std::vector< std::complex<double> > work( lwork );
    std::vector< double > rwork( lrwork );
    std::vector< lapack_int > iwork( liwork );
    return LAPACKE_zhbgvd_work(
        LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb,
        (lapack_complex_double*) AB, ldab,
        (lapack_complex_double*) BB, ldbb,
        W,
        (lapack_complex_double*) Z, ldz,
        (lapack_complex_double*) &work[0], lwork,
        &rwork[0], lrwork, &iwork[0], liwork );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hbgvx(
    char jobz, char range, char uplo, lapack_int n, lapack_int ka, lapack_int kb,
    float* AB, lapack_int ldab,
    float* BB, lapack_int ldbb,
    float* Q, lapack_int ldq,
    float vl, float vu, lapack_int il, lapack_int iu, float abstol,
    lapack_int* nfound, float* W,
    float* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_ssbgvx(
        LAPACK_COL_MAJOR, jobz, range, uplo, n, ka, kb,
        AB, ldab,
        BB, ldbb,
        Q, ldq, vl, vu, il, iu, abstol,
        nfound, W,
        Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_hbgvx(
    char jobz, char range, char uplo, lapack_int n, lapack_int ka, lapack_int kb,
    double* AB, lapack_int ldab,
    double* BB, lapack_int ldbb,
    double* Q, lapack_int ldq,
    double vl, double vu, lapack_int il, lapack_int iu, double abstol,
    lapack_int* nfound, double* W,
    double* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_dsbgvx(
        LAPACK_COL_MAJOR, jobz, range, uplo, n, ka, kb,
        AB, ldab,
        BB, ldbb,
        Q, ldq, vl, vu, il, iu, abstol,
        nfound, W,
        Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_hbgvx(
    char jobz, char range, char uplo, lapack_int n, lapack_int ka, lapack_int kb,
    std::complex<float>* AB, lapack_int ldab,
    std::complex<float>* BB, lapack_int ldbb,
    std::complex<float>* Q, lapack_int ldq,
    float vl, float vu, lapack_int il, lapack_int iu, float abstol,
    lapack_int* nfound, float* W,
    std::complex<float>* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_chbgvx(
        LAPACK_COL_MAJOR, jobz, range, uplo, n, ka, kb,
        (lapack_complex_float*) AB, ldab,
        (lapack_complex_float*) BB, ldbb,
        (lapack_complex_float*) Q, ldq, vl, vu, il, iu, abstol,
        nfound, W,
        (lapack_complex_float*) Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_hbgvx(
    char jobz, char range, char uplo, lapack_int n, lapack_int ka, lapack_int kb,
    std::complex<double>* AB, lapack_int ldab,
    std::complex<double>* BB, lapack_int ldbb,
    std::complex<double>* Q, lapack_int ldq,
    double vl, double vu, lapack_int il, lapack_int iu, double abstol,
    lapack_int* nfound, double* W,
    std::complex<double>* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_zhbgvx(
        LAPACK_COL_MAJOR, jobz, range, uplo, n, ka, kb,
        (lapack_complex_double*) AB, ldab,
        (lapack_complex_double*) BB, ldbb,
        (lapack_complex_double*) Q, ldq, vl, vu, il, iu, abstol,
        nfound, W,
        (lapack_complex_double*) Z, ldz,
        ifail );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hecon(
    char uplo, lapack_int n,
    float* A, lapack_int lda,
    lapack_int* ipiv, float anorm,
    float* rcond )
{
    return LAPACKE_ssycon(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_hecon(
    char uplo, lapack_int n,
    double* A, lapack_int lda,
    lapack_int* ipiv, double anorm,
    double* rcond )
{
    return LAPACKE_dsycon(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_hecon(
    char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv, float anorm,
    float* rcond )
{
    return LAPACKE_checon(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) A, lda,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_hecon(
    char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv, double anorm,
    double* rcond )
{
    return LAPACKE_zhecon(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) A, lda,
        ipiv, anorm,
        rcond );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_heev(
    char jobz, char uplo, lapack_int n,
    float* A, lapack_int lda,
    float* W )
{
    return LAPACKE_ssyev(
        LAPACK_COL_MAJOR, jobz, uplo, n,
        A, lda,
        W );
}

inline lapack_int LAPACKE_heev(
    char jobz, char uplo, lapack_int n,
    double* A, lapack_int lda,
    double* W )
{
    return LAPACKE_dsyev(
        LAPACK_COL_MAJOR, jobz, uplo, n,
        A, lda,
        W );
}

inline lapack_int LAPACKE_heev(
    char jobz, char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    float* W )
{
    return LAPACKE_cheev(
        LAPACK_COL_MAJOR, jobz, uplo, n,
        (lapack_complex_float*) A, lda,
        W );
}

inline lapack_int LAPACKE_heev(
    char jobz, char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    double* W )
{
    return LAPACKE_zheev(
        LAPACK_COL_MAJOR, jobz, uplo, n,
        (lapack_complex_double*) A, lda,
        W );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_heevd(
    char jobz, char uplo, lapack_int n,
    float* A, lapack_int lda,
    float* W )
{
    return LAPACKE_ssyevd(
        LAPACK_COL_MAJOR, jobz, uplo, n,
        A, lda,
        W );
}

inline lapack_int LAPACKE_heevd(
    char jobz, char uplo, lapack_int n,
    double* A, lapack_int lda,
    double* W )
{
    return LAPACKE_dsyevd(
        LAPACK_COL_MAJOR, jobz, uplo, n,
        A, lda,
        W );
}

inline lapack_int LAPACKE_heevd(
    char jobz, char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    float* W )
{
    return LAPACKE_cheevd(
        LAPACK_COL_MAJOR, jobz, uplo, n,
        (lapack_complex_float*) A, lda,
        W );
}

inline lapack_int LAPACKE_heevd(
    char jobz, char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    double* W )
{
    return LAPACKE_zheevd(
        LAPACK_COL_MAJOR, jobz, uplo, n,
        (lapack_complex_double*) A, lda,
        W );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_heevr(
    char jobz, char range, char uplo, lapack_int n,
    float* A, lapack_int lda,
    float vl, float vu, lapack_int il, lapack_int iu, float abstol,
    lapack_int* nfound, float* W,
    float* Z, lapack_int ldz,
    lapack_int* isuppz )
{
    return LAPACKE_ssyevr(
        LAPACK_COL_MAJOR, jobz, range, uplo, n,
        A, lda, vl, vu, il, iu, abstol,
        nfound, W,
        Z, ldz,
        isuppz );
}

inline lapack_int LAPACKE_heevr(
    char jobz, char range, char uplo, lapack_int n,
    double* A, lapack_int lda,
    double vl, double vu, lapack_int il, lapack_int iu, double abstol,
    lapack_int* nfound, double* W,
    double* Z, lapack_int ldz,
    lapack_int* isuppz )
{
    return LAPACKE_dsyevr(
        LAPACK_COL_MAJOR, jobz, range, uplo, n,
        A, lda, vl, vu, il, iu, abstol,
        nfound, W,
        Z, ldz,
        isuppz );
}

inline lapack_int LAPACKE_heevr(
    char jobz, char range, char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    float vl, float vu, lapack_int il, lapack_int iu, float abstol,
    lapack_int* nfound, float* W,
    std::complex<float>* Z, lapack_int ldz,
    lapack_int* isuppz )
{
    return LAPACKE_cheevr(
        LAPACK_COL_MAJOR, jobz, range, uplo, n,
        (lapack_complex_float*) A, lda, vl, vu, il, iu, abstol,
        nfound, W,
        (lapack_complex_float*) Z, ldz,
        isuppz );
}

inline lapack_int LAPACKE_heevr(
    char jobz, char range, char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    double vl, double vu, lapack_int il, lapack_int iu, double abstol,
    lapack_int* nfound, double* W,
    std::complex<double>* Z, lapack_int ldz,
    lapack_int* isuppz )
{
    return LAPACKE_zheevr(
        LAPACK_COL_MAJOR, jobz, range, uplo, n,
        (lapack_complex_double*) A, lda, vl, vu, il, iu, abstol,
        nfound, W,
        (lapack_complex_double*) Z, ldz,
        isuppz );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_heevx(
    char jobz, char range, char uplo, lapack_int n,
    float* A, lapack_int lda,
    float vl, float vu, lapack_int il, lapack_int iu, float abstol,
    lapack_int* nfound, float* W,
    float* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_ssyevx(
        LAPACK_COL_MAJOR, jobz, range, uplo, n,
        A, lda, vl, vu, il, iu, abstol,
        nfound, W,
        Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_heevx(
    char jobz, char range, char uplo, lapack_int n,
    double* A, lapack_int lda,
    double vl, double vu, lapack_int il, lapack_int iu, double abstol,
    lapack_int* nfound, double* W,
    double* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_dsyevx(
        LAPACK_COL_MAJOR, jobz, range, uplo, n,
        A, lda, vl, vu, il, iu, abstol,
        nfound, W,
        Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_heevx(
    char jobz, char range, char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    float vl, float vu, lapack_int il, lapack_int iu, float abstol,
    lapack_int* nfound, float* W,
    std::complex<float>* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_cheevx(
        LAPACK_COL_MAJOR, jobz, range, uplo, n,
        (lapack_complex_float*) A, lda, vl, vu, il, iu, abstol,
        nfound, W,
        (lapack_complex_float*) Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_heevx(
    char jobz, char range, char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    double vl, double vu, lapack_int il, lapack_int iu, double abstol,
    lapack_int* nfound, double* W,
    std::complex<double>* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_zheevx(
        LAPACK_COL_MAJOR, jobz, range, uplo, n,
        (lapack_complex_double*) A, lda, vl, vu, il, iu, abstol,
        nfound, W,
        (lapack_complex_double*) Z, ldz,
        ifail );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hegst(
    lapack_int itype, char uplo, lapack_int n,
    float* A, lapack_int lda,
    float* B, lapack_int ldb )
{
    return LAPACKE_ssygst(
        LAPACK_COL_MAJOR, itype, uplo, n,
        A, lda,
        B, ldb );
}

inline lapack_int LAPACKE_hegst(
    lapack_int itype, char uplo, lapack_int n,
    double* A, lapack_int lda,
    double* B, lapack_int ldb )
{
    return LAPACKE_dsygst(
        LAPACK_COL_MAJOR, itype, uplo, n,
        A, lda,
        B, ldb );
}

inline lapack_int LAPACKE_hegst(
    lapack_int itype, char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_chegst(
        LAPACK_COL_MAJOR, itype, uplo, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_hegst(
    lapack_int itype, char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zhegst(
        LAPACK_COL_MAJOR, itype, uplo, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hegv(
    lapack_int itype, char jobz, char uplo, lapack_int n,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    float* W )
{
    return LAPACKE_ssygv(
        LAPACK_COL_MAJOR, itype, jobz, uplo, n,
        A, lda,
        B, ldb,
        W );
}

inline lapack_int LAPACKE_hegv(
    lapack_int itype, char jobz, char uplo, lapack_int n,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    double* W )
{
    return LAPACKE_dsygv(
        LAPACK_COL_MAJOR, itype, jobz, uplo, n,
        A, lda,
        B, ldb,
        W );
}

inline lapack_int LAPACKE_hegv(
    lapack_int itype, char jobz, char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    float* W )
{
    return LAPACKE_chegv(
        LAPACK_COL_MAJOR, itype, jobz, uplo, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb,
        W );
}

inline lapack_int LAPACKE_hegv(
    lapack_int itype, char jobz, char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    double* W )
{
    return LAPACKE_zhegv(
        LAPACK_COL_MAJOR, itype, jobz, uplo, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb,
        W );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hegvd(
    lapack_int itype, char jobz, char uplo, lapack_int n,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    float* W )
{
    return LAPACKE_ssygvd(
        LAPACK_COL_MAJOR, itype, jobz, uplo, n,
        A, lda,
        B, ldb,
        W );
}

inline lapack_int LAPACKE_hegvd(
    lapack_int itype, char jobz, char uplo, lapack_int n,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    double* W )
{
    return LAPACKE_dsygvd(
        LAPACK_COL_MAJOR, itype, jobz, uplo, n,
        A, lda,
        B, ldb,
        W );
}

inline lapack_int LAPACKE_hegvd(
    lapack_int itype, char jobz, char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    float* W )
{
    return LAPACKE_chegvd(
        LAPACK_COL_MAJOR, itype, jobz, uplo, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb,
        W );
}

inline lapack_int LAPACKE_hegvd(
    lapack_int itype, char jobz, char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    double* W )
{
    return LAPACKE_zhegvd(
        LAPACK_COL_MAJOR, itype, jobz, uplo, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb,
        W );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hegvx(
    lapack_int itype, char jobz, char range, char uplo, lapack_int n,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    float vl, float vu, lapack_int il, lapack_int iu, float abstol,
    lapack_int* nfound, float* W,
    float* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_ssygvx(
        LAPACK_COL_MAJOR, itype, jobz, range, uplo, n,
        A, lda,
        B, ldb, vl, vu, il, iu, abstol,
        nfound, W,
        Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_hegvx(
    lapack_int itype, char jobz, char range, char uplo, lapack_int n,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    double vl, double vu, lapack_int il, lapack_int iu, double abstol,
    lapack_int* nfound, double* W,
    double* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_dsygvx(
        LAPACK_COL_MAJOR, itype, jobz, range, uplo, n,
        A, lda,
        B, ldb, vl, vu, il, iu, abstol,
        nfound, W,
        Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_hegvx(
    lapack_int itype, char jobz, char range, char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    float vl, float vu, lapack_int il, lapack_int iu, float abstol,
    lapack_int* nfound, float* W,
    std::complex<float>* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_chegvx(
        LAPACK_COL_MAJOR, itype, jobz, range, uplo, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb, vl, vu, il, iu, abstol,
        nfound, W,
        (lapack_complex_float*) Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_hegvx(
    lapack_int itype, char jobz, char range, char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    double vl, double vu, lapack_int il, lapack_int iu, double abstol,
    lapack_int* nfound, double* W,
    std::complex<double>* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_zhegvx(
        LAPACK_COL_MAJOR, itype, jobz, range, uplo, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb, vl, vu, il, iu, abstol,
        nfound, W,
        (lapack_complex_double*) Z, ldz,
        ifail );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_herfs(
    char uplo, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    float* AF, lapack_int ldaf,
    lapack_int* ipiv,
    float* B, lapack_int ldb,
    float* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_ssyrfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        AF, ldaf,
        ipiv,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_herfs(
    char uplo, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    double* AF, lapack_int ldaf,
    lapack_int* ipiv,
    double* B, lapack_int ldb,
    double* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_dsyrfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        AF, ldaf,
        ipiv,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_herfs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* AF, lapack_int ldaf,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_cherfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) AF, ldaf,
        ipiv,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_herfs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* AF, lapack_int ldaf,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_zherfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) AF, ldaf,
        ipiv,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) X, ldx,
        ferr,
        berr );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hesv(
    char uplo, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_ssysv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_hesv(
    char uplo, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dsysv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_hesv(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_chesv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) A, lda,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_hesv(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zhesv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) A, lda,
        ipiv,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hetrd(
    char uplo, lapack_int n,
    float* A, lapack_int lda,
    float* D,
    float* E,
    float* tau )
{
    return LAPACKE_ssytrd(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        D,
        E,
        tau );
}

inline lapack_int LAPACKE_hetrd(
    char uplo, lapack_int n,
    double* A, lapack_int lda,
    double* D,
    double* E,
    double* tau )
{
    return LAPACKE_dsytrd(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        D,
        E,
        tau );
}

inline lapack_int LAPACKE_hetrd(
    char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    float* D,
    float* E,
    std::complex<float>* tau )
{
    return LAPACKE_chetrd(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) A, lda,
        D,
        E,
        (lapack_complex_float*) tau );
}

inline lapack_int LAPACKE_hetrd(
    char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    double* D,
    double* E,
    std::complex<double>* tau )
{
    return LAPACKE_zhetrd(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) A, lda,
        D,
        E,
        (lapack_complex_double*) tau );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hetrf(
    char uplo, lapack_int n,
    float* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_ssytrf(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        ipiv );
}

inline lapack_int LAPACKE_hetrf(
    char uplo, lapack_int n,
    double* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_dsytrf(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        ipiv );
}

inline lapack_int LAPACKE_hetrf(
    char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_chetrf(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) A, lda,
        ipiv );
}

inline lapack_int LAPACKE_hetrf(
    char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_zhetrf(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) A, lda,
        ipiv );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hetri(
    char uplo, lapack_int n,
    float* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_ssytri(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        ipiv );
}

inline lapack_int LAPACKE_hetri(
    char uplo, lapack_int n,
    double* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_dsytri(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        ipiv );
}

inline lapack_int LAPACKE_hetri(
    char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_chetri(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) A, lda,
        ipiv );
}

inline lapack_int LAPACKE_hetri(
    char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_zhetri(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) A, lda,
        ipiv );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hetrs(
    char uplo, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_ssytrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_hetrs(
    char uplo, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dsytrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_hetrs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_chetrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) A, lda,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_hetrs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zhetrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) A, lda,
        ipiv,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hpcon(
    char uplo, lapack_int n,
    float* AP,
    lapack_int* ipiv, float anorm,
    float* rcond )
{
    return LAPACKE_sspcon(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_hpcon(
    char uplo, lapack_int n,
    double* AP,
    lapack_int* ipiv, double anorm,
    double* rcond )
{
    return LAPACKE_dspcon(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_hpcon(
    char uplo, lapack_int n,
    std::complex<float>* AP,
    lapack_int* ipiv, float anorm,
    float* rcond )
{
    return LAPACKE_chpcon(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) AP,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_hpcon(
    char uplo, lapack_int n,
    std::complex<double>* AP,
    lapack_int* ipiv, double anorm,
    double* rcond )
{
    return LAPACKE_zhpcon(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) AP,
        ipiv, anorm,
        rcond );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hpev(
    char jobz, char uplo, lapack_int n,
    float* AP,
    float* W,
    float* Z, lapack_int ldz )
{
    return LAPACKE_sspev(
        LAPACK_COL_MAJOR, jobz, uplo, n,
        AP,
        W,
        Z, ldz );
}

inline lapack_int LAPACKE_hpev(
    char jobz, char uplo, lapack_int n,
    double* AP,
    double* W,
    double* Z, lapack_int ldz )
{
    return LAPACKE_dspev(
        LAPACK_COL_MAJOR, jobz, uplo, n,
        AP,
        W,
        Z, ldz );
}

inline lapack_int LAPACKE_hpev(
    char jobz, char uplo, lapack_int n,
    std::complex<float>* AP,
    float* W,
    std::complex<float>* Z, lapack_int ldz )
{
    return LAPACKE_chpev(
        LAPACK_COL_MAJOR, jobz, uplo, n,
        (lapack_complex_float*) AP,
        W,
        (lapack_complex_float*) Z, ldz );
}

inline lapack_int LAPACKE_hpev(
    char jobz, char uplo, lapack_int n,
    std::complex<double>* AP,
    double* W,
    std::complex<double>* Z, lapack_int ldz )
{
    return LAPACKE_zhpev(
        LAPACK_COL_MAJOR, jobz, uplo, n,
        (lapack_complex_double*) AP,
        W,
        (lapack_complex_double*) Z, ldz );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hpevd(
    char jobz, char uplo, lapack_int n,
    float* AP,
    float* W,
    float* Z, lapack_int ldz )
{
    return LAPACKE_sspevd(
        LAPACK_COL_MAJOR, jobz, uplo, n,
        AP,
        W,
        Z, ldz );
}

inline lapack_int LAPACKE_hpevd(
    char jobz, char uplo, lapack_int n,
    double* AP,
    double* W,
    double* Z, lapack_int ldz )
{
    return LAPACKE_dspevd(
        LAPACK_COL_MAJOR, jobz, uplo, n,
        AP,
        W,
        Z, ldz );
}

inline lapack_int LAPACKE_hpevd(
    char jobz, char uplo, lapack_int n,
    std::complex<float>* AP,
    float* W,
    std::complex<float>* Z, lapack_int ldz )
{
    return LAPACKE_chpevd(
        LAPACK_COL_MAJOR, jobz, uplo, n,
        (lapack_complex_float*) AP,
        W,
        (lapack_complex_float*) Z, ldz );
}

inline lapack_int LAPACKE_hpevd(
    char jobz, char uplo, lapack_int n,
    std::complex<double>* AP,
    double* W,
    std::complex<double>* Z, lapack_int ldz )
{
    return LAPACKE_zhpevd(
        LAPACK_COL_MAJOR, jobz, uplo, n,
        (lapack_complex_double*) AP,
        W,
        (lapack_complex_double*) Z, ldz );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hpevx(
    char jobz, char range, char uplo, lapack_int n,
    float* AP,
    float vl, float vu, lapack_int il, lapack_int iu, float abstol,
    lapack_int* nfound, float* W,
    float* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_sspevx(
        LAPACK_COL_MAJOR, jobz, range, uplo, n,
        AP, vl, vu, il, iu, abstol,
        nfound,
        W,
        Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_hpevx(
    char jobz, char range, char uplo, lapack_int n,
    double* AP,
    double vl, double vu, lapack_int il, lapack_int iu, double abstol,
    lapack_int* nfound, double* W,
    double* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_dspevx(
        LAPACK_COL_MAJOR, jobz, range, uplo, n,
        AP, vl, vu, il, iu, abstol,
        nfound,
        W,
        Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_hpevx(
    char jobz, char range, char uplo, lapack_int n,
    std::complex<float>* AP,
    float vl, float vu, lapack_int il, lapack_int iu, float abstol,
    lapack_int* nfound, float* W,
    std::complex<float>* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_chpevx(
        LAPACK_COL_MAJOR, jobz, range, uplo, n,
        (lapack_complex_float*) AP, vl, vu, il, iu, abstol,
        nfound,
        W,
        (lapack_complex_float*) Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_hpevx(
    char jobz, char range, char uplo, lapack_int n,
    std::complex<double>* AP,
    double vl, double vu, lapack_int il, lapack_int iu, double abstol,
    lapack_int* nfound, double* W,
    std::complex<double>* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_zhpevx(
        LAPACK_COL_MAJOR, jobz, range, uplo, n,
        (lapack_complex_double*) AP, vl, vu, il, iu, abstol,
        nfound,
        W,
        (lapack_complex_double*) Z, ldz,
        ifail );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hpgst(
    lapack_int itype, char uplo, lapack_int n,
    float* AP,
    float* BP )
{
    return LAPACKE_sspgst(
        LAPACK_COL_MAJOR, itype, uplo, n,
        AP,
        BP );
}

inline lapack_int LAPACKE_hpgst(
    lapack_int itype, char uplo, lapack_int n,
    double* AP,
    double* BP )
{
    return LAPACKE_dspgst(
        LAPACK_COL_MAJOR, itype, uplo, n,
        AP,
        BP );
}

inline lapack_int LAPACKE_hpgst(
    lapack_int itype, char uplo, lapack_int n,
    std::complex<float>* AP,
    std::complex<float>* BP )
{
    return LAPACKE_chpgst(
        LAPACK_COL_MAJOR, itype, uplo, n,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) BP );
}

inline lapack_int LAPACKE_hpgst(
    lapack_int itype, char uplo, lapack_int n,
    std::complex<double>* AP,
    std::complex<double>* BP )
{
    return LAPACKE_zhpgst(
        LAPACK_COL_MAJOR, itype, uplo, n,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) BP );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hpgv(
    lapack_int itype, char jobz, char uplo, lapack_int n,
    float* AP,
    float* BP,
    float* W,
    float* Z, lapack_int ldz )
{
    return LAPACKE_sspgv(
        LAPACK_COL_MAJOR, itype, jobz, uplo, n,
        AP,
        BP,
        W,
        Z, ldz );
}

inline lapack_int LAPACKE_hpgv(
    lapack_int itype, char jobz, char uplo, lapack_int n,
    double* AP,
    double* BP,
    double* W,
    double* Z, lapack_int ldz )
{
    return LAPACKE_dspgv(
        LAPACK_COL_MAJOR, itype, jobz, uplo, n,
        AP,
        BP,
        W,
        Z, ldz );
}

inline lapack_int LAPACKE_hpgv(
    lapack_int itype, char jobz, char uplo, lapack_int n,
    std::complex<float>* AP,
    std::complex<float>* BP,
    float* W,
    std::complex<float>* Z, lapack_int ldz )
{
    return LAPACKE_chpgv(
        LAPACK_COL_MAJOR, itype, jobz, uplo, n,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) BP,
        W,
        (lapack_complex_float*) Z, ldz );
}

inline lapack_int LAPACKE_hpgv(
    lapack_int itype, char jobz, char uplo, lapack_int n,
    std::complex<double>* AP,
    std::complex<double>* BP,
    double* W,
    std::complex<double>* Z, lapack_int ldz )
{
    return LAPACKE_zhpgv(
        LAPACK_COL_MAJOR, itype, jobz, uplo, n,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) BP,
        W,
        (lapack_complex_double*) Z, ldz );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hpgvd(
    lapack_int itype, char jobz, char uplo, lapack_int n,
    float* AP,
    float* BP,
    float* W,
    float* Z, lapack_int ldz )
{
    return LAPACKE_sspgvd(
        LAPACK_COL_MAJOR, itype, jobz, uplo, n,
        AP,
        BP,
        W,
        Z, ldz );
}

inline lapack_int LAPACKE_hpgvd(
    lapack_int itype, char jobz, char uplo, lapack_int n,
    double* AP,
    double* BP,
    double* W,
    double* Z, lapack_int ldz )
{
    return LAPACKE_dspgvd(
        LAPACK_COL_MAJOR, itype, jobz, uplo, n,
        AP,
        BP,
        W,
        Z, ldz );
}

inline lapack_int LAPACKE_hpgvd(
    lapack_int itype, char jobz, char uplo, lapack_int n,
    std::complex<float>* AP,
    std::complex<float>* BP,
    float* W,
    std::complex<float>* Z, lapack_int ldz )
{
    return LAPACKE_chpgvd(
        LAPACK_COL_MAJOR, itype, jobz, uplo, n,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) BP,
        W,
        (lapack_complex_float*) Z, ldz );
}

inline lapack_int LAPACKE_hpgvd(
    lapack_int itype, char jobz, char uplo, lapack_int n,
    std::complex<double>* AP,
    std::complex<double>* BP,
    double* W,
    std::complex<double>* Z, lapack_int ldz )
{
    return LAPACKE_zhpgvd(
        LAPACK_COL_MAJOR, itype, jobz, uplo, n,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) BP,
        W,
        (lapack_complex_double*) Z, ldz );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hpgvx(
    lapack_int itype, char jobz, char range, char uplo, lapack_int n,
    float* AP,
    float* BP,
    float vl, float vu, lapack_int il, lapack_int iu, float abstol,
    lapack_int* nfound, float* W,
    float* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_sspgvx(
        LAPACK_COL_MAJOR, itype, jobz, range, uplo, n,
        AP,
        BP, vl, vu, il, iu, abstol,
        nfound, W,
        Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_hpgvx(
    lapack_int itype, char jobz, char range, char uplo, lapack_int n,
    double* AP,
    double* BP,
    double vl, double vu, lapack_int il, lapack_int iu, double abstol,
    lapack_int* nfound, double* W,
    double* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_dspgvx(
        LAPACK_COL_MAJOR, itype, jobz, range, uplo, n,
        AP,
        BP, vl, vu, il, iu, abstol,
        nfound, W,
        Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_hpgvx(
    lapack_int itype, char jobz, char range, char uplo, lapack_int n,
    std::complex<float>* AP,
    std::complex<float>* BP,
    float vl, float vu, lapack_int il, lapack_int iu, float abstol,
    lapack_int* nfound, float* W,
    std::complex<float>* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_chpgvx(
        LAPACK_COL_MAJOR, itype, jobz, range, uplo, n,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) BP, vl, vu, il, iu, abstol,
        nfound, W,
        (lapack_complex_float*) Z, ldz,
        ifail );
}

inline lapack_int LAPACKE_hpgvx(
    lapack_int itype, char jobz, char range, char uplo, lapack_int n,
    std::complex<double>* AP,
    std::complex<double>* BP,
    double vl, double vu, lapack_int il, lapack_int iu, double abstol,
    lapack_int* nfound, double* W,
    std::complex<double>* Z, lapack_int ldz,
    lapack_int* ifail )
{
    return LAPACKE_zhpgvx(
        LAPACK_COL_MAJOR, itype, jobz, range, uplo, n,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) BP, vl, vu, il, iu, abstol,
        nfound, W,
        (lapack_complex_double*) Z, ldz,
        ifail );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hprfs(
    char uplo, lapack_int n, lapack_int nrhs,
    float* AP,
    float* AFP,
    lapack_int* ipiv,
    float* B, lapack_int ldb,
    float* X, lapack_int ldx,
    float* ferr, float* berr )
{
    return LAPACKE_ssprfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        AFP,
        ipiv,
        B, ldb,
        X, ldx,
        ferr, berr );
}

inline lapack_int LAPACKE_hprfs(
    char uplo, lapack_int n, lapack_int nrhs,
    double* AP,
    double* AFP,
    lapack_int* ipiv,
    double* B, lapack_int ldb,
    double* X, lapack_int ldx,
    double* ferr, double* berr )
{
    return LAPACKE_dsprfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        AFP,
        ipiv,
        B, ldb,
        X, ldx,
        ferr, berr );
}

inline lapack_int LAPACKE_hprfs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* AP,
    std::complex<float>* AFP,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* X, lapack_int ldx,
    float* ferr, float* berr )
{
    return LAPACKE_chprfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) AFP,
        ipiv,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) X, ldx,
        ferr, berr );
}

inline lapack_int LAPACKE_hprfs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* AP,
    std::complex<double>* AFP,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* X, lapack_int ldx,
    double* ferr, double* berr )
{
    return LAPACKE_zhprfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) AFP,
        ipiv,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) X, ldx,
        ferr, berr );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hpsv(
    char uplo, lapack_int n, lapack_int nrhs,
    float* AP,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_sspsv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_hpsv(
    char uplo, lapack_int n, lapack_int nrhs,
    double* AP,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dspsv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_hpsv(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* AP,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_chpsv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) AP,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_hpsv(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* AP,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zhpsv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) AP,
        ipiv,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hptrd(
    char uplo, lapack_int n,
    float* AP,
    float* D, float* E,
    float* tau )
{
    return LAPACKE_ssptrd(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        D, E,
        tau );
}

inline lapack_int LAPACKE_hptrd(
    char uplo, lapack_int n,
    double* AP,
    double* D, double* E,
    double* tau )
{
    return LAPACKE_dsptrd(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        D, E,
        tau );
}

inline lapack_int LAPACKE_hptrd(
    char uplo, lapack_int n,
    std::complex<float>* AP,
    float* D, float* E,
    std::complex<float>* tau )
{
    return LAPACKE_chptrd(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) AP,
        D, E,
        (lapack_complex_float*) tau );
}

inline lapack_int LAPACKE_hptrd(
    char uplo, lapack_int n,
    std::complex<double>* AP,
    double* D, double* E,
    std::complex<double>* tau )
{
    return LAPACKE_zhptrd(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) AP,
        D, E,
        (lapack_complex_double*) tau );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hptrf(
    char uplo, lapack_int n,
    float* AP,
    lapack_int* ipiv )
{
    return LAPACKE_ssptrf(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        ipiv );
}

inline lapack_int LAPACKE_hptrf(
    char uplo, lapack_int n,
    double* AP,
    lapack_int* ipiv )
{
    return LAPACKE_dsptrf(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        ipiv );
}

inline lapack_int LAPACKE_hptrf(
    char uplo, lapack_int n,
    std::complex<float>* AP,
    lapack_int* ipiv )
{
    return LAPACKE_chptrf(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) AP,
        ipiv );
}

inline lapack_int LAPACKE_hptrf(
    char uplo, lapack_int n,
    std::complex<double>* AP,
    lapack_int* ipiv )
{
    return LAPACKE_zhptrf(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) AP,
        ipiv );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hptri(
    char uplo, lapack_int n,
    float* AP,
    lapack_int* ipiv )
{
    return LAPACKE_ssptri(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        ipiv );
}

inline lapack_int LAPACKE_hptri(
    char uplo, lapack_int n,
    double* AP,
    lapack_int* ipiv )
{
    return LAPACKE_dsptri(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        ipiv );
}

inline lapack_int LAPACKE_hptri(
    char uplo, lapack_int n,
    std::complex<float>* AP,
    lapack_int* ipiv )
{
    return LAPACKE_chptri(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) AP,
        ipiv );
}

inline lapack_int LAPACKE_hptri(
    char uplo, lapack_int n,
    std::complex<double>* AP,
    lapack_int* ipiv )
{
    return LAPACKE_zhptri(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) AP,
        ipiv );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_hptrs(
    char uplo, lapack_int n, lapack_int nrhs,
    float* AP,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_ssptrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_hptrs(
    char uplo, lapack_int n, lapack_int nrhs,
    double* AP,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dsptrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_hptrs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* AP,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_chptrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) AP,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_hptrs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* AP,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zhptrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) AP,
        ipiv,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_lacpy(
    char uplo, lapack_int m, lapack_int n,
    float* A, lapack_int lda,
    float* B, lapack_int ldb )
{
    return LAPACKE_slacpy(
        LAPACK_COL_MAJOR, uplo, m, n,
        A, lda,
        B, ldb );
}

inline lapack_int LAPACKE_lacpy(
    char uplo, lapack_int m, lapack_int n,
    double* A, lapack_int lda,
    double* B, lapack_int ldb )
{
    return LAPACKE_dlacpy(
        LAPACK_COL_MAJOR, uplo, m, n,
        A, lda,
        B, ldb );
}

inline lapack_int LAPACKE_lacpy(
    char uplo, lapack_int m, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_clacpy(
        LAPACK_COL_MAJOR, uplo, m, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_lacpy(
    char uplo, lapack_int m, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zlacpy(
        LAPACK_COL_MAJOR, uplo, m, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
// Fortran prototypes if not given via lapacke.h
extern "C" {

/* ----- matrix norm - general banded */
#ifndef LAPACK_slaed4
#define LAPACK_slaed4 LAPACK_GLOBAL(slaed4,SLAED4)
void LAPACK_slaed4(
    lapack_int const* n, lapack_int const* i,
    float const* d, float const* z,
    float* delta, float const* rho, float* lambda, lapack_int* info );
#endif

#ifndef LAPACK_dlaed4
#define LAPACK_dlaed4 LAPACK_GLOBAL(dlaed4,DLAED4)
void LAPACK_dlaed4(
    lapack_int const* n, lapack_int const* i,
    double const* d, double const* z,
    double* delta, double const* rho, double* lambda, lapack_int* info );
#endif

}  // extern "C"

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_laed4(
    lapack_int n, lapack_int i,
    float const* d, float const* z,
    float* delta, float rho, float* lambda )
{
    i += 1;  // change to 1-based
    lapack_int info = 0;
    LAPACK_slaed4( &n, &i, d, z, delta, &rho, lambda, &info );
    return info;

    //return LAPACKE_slaed4( n, i, d, z, delta, rho, lambda );
}

inline lapack_int LAPACKE_laed4(
    lapack_int n, lapack_int i,
    double const* d, double const* z,
    double* delta, double rho, double* lambda )
{
    i += 1;  // change to 1-based
    lapack_int info = 0;
    LAPACK_dlaed4( &n, &i, d, z, delta, &rho, lambda, &info );
    return info;

    //return LAPACKE_dlaed4( n, i, d, z, delta, rho, lambda );
}

// -----------------------------------------------------------------------------
// Fortran prototypes if not given via lapacke.h
extern "C" {

/* ----- matrix norm - general banded */
#ifndef LAPACK_slangb
#define LAPACK_slangb LAPACK_GLOBAL(slangb,SLANGB)
lapack_float_return LAPACK_slangb(
    char const* norm, lapack_int const* n,
    lapack_int const* kl, lapack_int const* ku,
    float const* AB, lapack_int const* ldab,
    float* work );
#endif

#ifndef LAPACK_dlangb
#define LAPACK_dlangb LAPACK_GLOBAL(dlangb,DLANGB)
double LAPACK_dlangb(
    char const* norm, lapack_int const* n,
    lapack_int const* kl, lapack_int const* ku,
    double const* AB, lapack_int const* ldab,
    double* work );
#endif

#ifndef LAPACK_clangb
#define LAPACK_clangb LAPACK_GLOBAL(clangb,CLANGB)
lapack_float_return LAPACK_clangb(
    char const* norm, lapack_int const* n,
    lapack_int const* kl, lapack_int const* ku,
    lapack_complex_float const* AB, lapack_int const* ldab,
    float* work );
#endif

#ifndef LAPACK_zlangb
#define LAPACK_zlangb LAPACK_GLOBAL(zlangb,ZLANGB)
double LAPACK_zlangb(
    char const* norm, lapack_int const* n,
    lapack_int const* kl, lapack_int const* ku,
    lapack_complex_double const* AB, lapack_int const* ldab,
    double* work );
#endif

}  // extern "C"

// --------------------
// wrappers around LAPACK (not in LAPACKE)
inline float LAPACKE_langb(
    char norm, lapack_int n, lapack_int kl, lapack_int ku,
    float* AB, lapack_int ldab )
{
    std::vector< float > work( n );
    return LAPACK_slangb(
        &norm, &n, &kl, &ku,
        AB, &ldab, &work[0] );
}

inline double LAPACKE_langb(
    char norm, lapack_int n, lapack_int kl, lapack_int ku,
    double* AB, lapack_int ldab )
{
    std::vector< double > work( n );
    return LAPACK_dlangb(
        &norm, &n, &kl, &ku,
        AB, &ldab, &work[0] );
}

inline float LAPACKE_langb(
    char norm, lapack_int n, lapack_int kl, lapack_int ku,
    std::complex<float>* AB, lapack_int ldab )
{
    std::vector< float > work( n );
    return LAPACK_clangb(
        &norm, &n, &kl, &ku,
        (lapack_complex_float*) AB, &ldab, &work[0] );
}

inline double LAPACKE_langb(
    char norm, lapack_int n, lapack_int kl, lapack_int ku,
    std::complex<double>* AB, lapack_int ldab )
{
    std::vector< double > work( n );
    return LAPACK_zlangb(
        &norm, &n, &kl, &ku,
        (lapack_complex_double*) AB, &ldab, &work[0] );
}

// -----------------------------------------------------------------------------
inline float LAPACKE_lange(
    char norm, lapack_int m, lapack_int n,
    float* A, lapack_int lda )
{
    return LAPACKE_slange(
        LAPACK_COL_MAJOR, norm, m, n,
        A, lda );
}

inline double LAPACKE_lange(
    char norm, lapack_int m, lapack_int n,
    double* A, lapack_int lda )
{
    return LAPACKE_dlange(
        LAPACK_COL_MAJOR, norm, m, n,
        A, lda );
}

inline float LAPACKE_lange(
    char norm, lapack_int m, lapack_int n,
    std::complex<float>* A, lapack_int lda )
{
    return LAPACKE_clange(
        LAPACK_COL_MAJOR, norm, m, n,
        (lapack_complex_float*) A, lda );
}

inline double LAPACKE_lange(
    char norm, lapack_int m, lapack_int n,
    std::complex<double>* A, lapack_int lda )
{
    return LAPACKE_zlange(
        LAPACK_COL_MAJOR, norm, m, n,
        (lapack_complex_double*) A, lda );
}

// -----------------------------------------------------------------------------
// Fortran prototypes if not given via lapacke.h
extern "C" {

/* ----- matrix norm - general tridiagonal */
#ifndef LAPACK_slangt
#define LAPACK_slangt LAPACK_GLOBAL(slangt,SLANGT)
lapack_float_return LAPACK_slangt(
    char const* norm, lapack_int const* n,
    float const* DL,
    float const* D,
    float const* DU );
#endif

#ifndef LAPACK_dlangt
#define LAPACK_dlangt LAPACK_GLOBAL(dlangt,DLANGT)
double LAPACK_dlangt(
    char const* norm, lapack_int const* n,
    double const* DL,
    double const* D,
    double const* DU );
#endif

#ifndef LAPACK_clangt
#define LAPACK_clangt LAPACK_GLOBAL(clangt,CLANGT)
lapack_float_return LAPACK_clangt(
    char const* norm, lapack_int const* n,
    lapack_complex_float const* DL,
    lapack_complex_float const* D,
    lapack_complex_float const* DU );
#endif

#ifndef LAPACK_zlangt
#define LAPACK_zlangt LAPACK_GLOBAL(zlangt,ZLANGT)
double LAPACK_zlangt(
    char const* norm, lapack_int const* n,
    lapack_complex_double const* DL,
    lapack_complex_double const* D,
    lapack_complex_double const* DU );
#endif

}  // extern "C"

// --------------------
// wrappers around LAPACK (not in LAPACKE)
inline float LAPACKE_langt(
    char norm, lapack_int n,
    float* DL,
    float* D,
    float* DU )
{
    return LAPACK_slangt(
        &norm, &n,
        DL,
        D,
        DU );
}

inline double LAPACKE_langt(
    char norm, lapack_int n,
    double* DL,
    double* D,
    double* DU )
{
    return LAPACK_dlangt(
        &norm, &n,
        DL,
        D,
        DU );
}

inline float LAPACKE_langt(
    char norm, lapack_int n,
    std::complex<float>* DL,
    std::complex<float>* D,
    std::complex<float>* DU )
{
    return LAPACK_clangt(
        &norm, &n,
        (lapack_complex_float*) DL,
        (lapack_complex_float*) D,
        (lapack_complex_float*) DU );
}

inline double LAPACKE_langt(
    char norm, lapack_int n,
    std::complex<double>* DL,
    std::complex<double>* D,
    std::complex<double>* DU )
{
    return LAPACK_zlangt(
        &norm, &n,
        (lapack_complex_double*) DL,
        (lapack_complex_double*) D,
        (lapack_complex_double*) DU );
}

// -----------------------------------------------------------------------------
// Fortran prototypes if not given via lapacke.h
extern "C" {

/* ----- matrix norm - symmetric banded */
#ifndef LAPACK_slansb
#define LAPACK_slansb LAPACK_GLOBAL(slansb,SLANSB)
lapack_float_return LAPACK_slansb(
    char const* norm, char const* uplo,
    lapack_int const* n, lapack_int const* kd,
    float const* AB, lapack_int const* ldab,
    float* work );
#endif

#ifndef LAPACK_dlansb
#define LAPACK_dlansb LAPACK_GLOBAL(dlansb,DLANSB)
double LAPACK_dlansb(
    char const* norm, char const* uplo,
    lapack_int const* n, lapack_int const* kd,
    double const* AB, lapack_int const* ldab,
    double* work );
#endif

#ifndef LAPACK_clanhb
#define LAPACK_clanhb LAPACK_GLOBAL(clanhb,CLANHB)
lapack_float_return LAPACK_clanhb(
    char const* norm, char const* uplo,
    lapack_int const* n, lapack_int const* kd,
    lapack_complex_float const* AB, lapack_int const* ldab,
    float* work );
#endif

#ifndef LAPACK_zlanhb
#define LAPACK_zlanhb LAPACK_GLOBAL(zlanhb,ZLANHB)
double LAPACK_zlanhb(
    char const* norm, char const* uplo,
    lapack_int const* n, lapack_int const* kd,
    lapack_complex_double const* AB, lapack_int const* ldab,
    double* work );
#endif

}  // extern "C"

// --------------------
// wrappers around LAPACK (not in LAPACKE)
inline float LAPACKE_lanhb(
    char norm, char uplo, lapack_int n, lapack_int kd,
    float* AB, lapack_int ldab )
{
    std::vector< float > work( n );
    return LAPACK_slansb(
        &norm, &uplo, &n, &kd,
        AB, &ldab, &work[0] );
}

inline double LAPACKE_lanhb(
    char norm, char uplo, lapack_int n, lapack_int kd,
    double* AB, lapack_int ldab )
{
    std::vector< double > work( n );
    return LAPACK_dlansb(
        &norm, &uplo, &n, &kd,
        AB, &ldab, &work[0] );
}

inline float LAPACKE_lanhb(
    char norm, char uplo, lapack_int n, lapack_int kd,
    std::complex<float>* AB, lapack_int ldab )
{
    std::vector< float > work( n );
    return LAPACK_clanhb(
        &norm, &uplo, &n, &kd,
        (lapack_complex_float*) AB, &ldab, &work[0] );
}

inline double LAPACKE_lanhb(
    char norm, char uplo, lapack_int n, lapack_int kd,
    std::complex<double>* AB, lapack_int ldab )
{
    std::vector< double > work( n );
    return LAPACK_zlanhb(
        &norm, &uplo, &n, &kd,
        (lapack_complex_double*) AB, &ldab, &work[0] );
}

// -----------------------------------------------------------------------------
inline float LAPACKE_lanhe(
    char norm, char uplo, lapack_int n,
    float* A, lapack_int lda )
{
    return LAPACKE_slansy(
        LAPACK_COL_MAJOR, norm, uplo, n,
        A, lda );
}

inline double LAPACKE_lanhe(
    char norm, char uplo, lapack_int n,
    double* A, lapack_int lda )
{
    return LAPACKE_dlansy(
        LAPACK_COL_MAJOR, norm, uplo, n,
        A, lda );
}

inline float LAPACKE_lanhe(
    char norm, char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda )
{
    return LAPACKE_clanhe(
        LAPACK_COL_MAJOR, norm, uplo, n,
        (lapack_complex_float*) A, lda );
}

inline double LAPACKE_lanhe(
    char norm, char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda )
{
    return LAPACKE_zlanhe(
        LAPACK_COL_MAJOR, norm, uplo, n,
        (lapack_complex_double*) A, lda );
}

// -----------------------------------------------------------------------------
// Fortran prototypes if not given via lapacke.h
extern "C" {

/* ----- matrix norm - symmetric packed */
#ifndef LAPACK_slansp
#define LAPACK_slansp LAPACK_GLOBAL(slansp,SLANSP)
lapack_float_return LAPACK_slansp(
    char const* norm, char const* uplo,
    lapack_int const* n,
    float const* AP, float* work );
#endif

#ifndef LAPACK_dlansp
#define LAPACK_dlansp LAPACK_GLOBAL(dlansp,DLANSP)
double LAPACK_dlansp(
    char const* norm, char const* uplo,
    lapack_int const* n,
    double const* AP, double* work );
#endif

#ifndef LAPACK_clanhp
#define LAPACK_clanhp LAPACK_GLOBAL(clanhp,CLANHP)
lapack_float_return LAPACK_clanhp(
    char const* norm, char const* uplo,
    lapack_int const* n,
    lapack_complex_float const* AP, float* work );
#endif

#ifndef LAPACK_zlanhp
#define LAPACK_zlanhp LAPACK_GLOBAL(zlanhp,ZLANHP)
double LAPACK_zlanhp(
    char const* norm, char const* uplo,
    lapack_int const* n,
    lapack_complex_double const* AP, double* work );
#endif

}  // extern "C"

// --------------------
// wrappers around LAPACK (not in LAPACKE)
inline float LAPACKE_lanhp(
    char norm, char uplo, lapack_int n,
    float* AP )
{
    std::vector< float > work( n );
    return LAPACK_slansp(
        &norm, &uplo, &n,
        AP, &work[0] );
}

inline double LAPACKE_lanhp(
    char norm, char uplo, lapack_int n,
    double* AP )
{
    std::vector< double > work( n );
    return LAPACK_dlansp(
        &norm, &uplo, &n,
        AP, &work[0] );
}

inline float LAPACKE_lanhp(
    char norm, char uplo, lapack_int n,
    std::complex<float>* AP )
{
    std::vector< float > work( n );
    return LAPACK_clanhp(
        &norm, &uplo, &n,
        (lapack_complex_float*) AP, &work[0] );
}

inline double LAPACKE_lanhp(
    char norm, char uplo, lapack_int n,
    std::complex<double>* AP )
{
    std::vector< double > work( n );
    return LAPACK_zlanhp(
        &norm, &uplo, &n,
        (lapack_complex_double*) AP, &work[0] );
}

// -----------------------------------------------------------------------------
// Fortran prototypes if not given via lapacke.h
extern "C" {

/* ----- matrix norm - symmetric packed */
#ifndef LAPACK_slanhs
#define LAPACK_slanhs LAPACK_GLOBAL(slanhs,SLANHS)
lapack_float_return LAPACK_slanhs(
    char const* norm,
    lapack_int const* n,
    float const* A, lapack_int const* lda, float* work );
#endif

#ifndef LAPACK_dlanhs
#define LAPACK_dlanhs LAPACK_GLOBAL(dlanhs,DLANHS)
double LAPACK_dlanhs(
    char const* norm,
    lapack_int const* n,
    double const* A, lapack_int const* lda, double* work );
#endif

#ifndef LAPACK_clanhs
#define LAPACK_clanhs LAPACK_GLOBAL(clanhs,CLANHS)
lapack_float_return LAPACK_clanhs(
    char const* norm,
    lapack_int const* n,
    lapack_complex_float const* A, lapack_int const* lda, float* work );
#endif

#ifndef LAPACK_zlanhs
#define LAPACK_zlanhs LAPACK_GLOBAL(zlanhs,ZLANHS)
double LAPACK_zlanhs(
    char const* norm,
    lapack_int const* n,
    lapack_complex_double const* A, lapack_int const* lda, double* work );
#endif

}  // extern "C"

// --------------------
// wrappers around LAPACK (not in LAPACKE)
static float LAPACKE_lanhs(
    char norm, lapack_int n,
    float* A, lapack_int lda )
{
    std::vector< float > work( n );
    return LAPACK_slanhs(
        &norm, &n,
        A, &lda, &work[0] );
}

static double LAPACKE_lanhs(
    char norm, lapack_int n,
    double* A, lapack_int lda )
{
    std::vector< double > work( n );
    return LAPACK_dlanhs(
        &norm, &n,
        A, &lda, &work[0] );
}

static float LAPACKE_lanhs(
    char norm, lapack_int n,
    std::complex<float>* A, lapack_int lda )
{
    std::vector< float > work( n );
    return LAPACK_clanhs(
        &norm, &n,
        (lapack_complex_float*) A, &lda, &work[0] );
}

static double LAPACKE_lanhs(
    char norm, lapack_int n,
    std::complex<double>* A, lapack_int lda )
{
    std::vector< double > work( n );
    return LAPACK_zlanhs(
        &norm, &n,
        (lapack_complex_double*) A, &lda, &work[0] );
}

// -----------------------------------------------------------------------------
// Fortran prototypes if not given via lapacke.h
extern "C" {

/* ----- matrix norm - symmetric tridiagonal */
#ifndef LAPACK_slanst
#define LAPACK_slanst LAPACK_GLOBAL(slanst,SLANST)
lapack_float_return LAPACK_slanst(
    char const* norm, lapack_int const* n,
    float const* D,
    float const* E );
#endif

#ifndef LAPACK_dlanst
#define LAPACK_dlanst LAPACK_GLOBAL(dlanst,DLANST)
double LAPACK_dlanst(
    char const* norm, lapack_int const* n,
    double const* D,
    double const* E );
#endif

#ifndef LAPACK_clanht
#define LAPACK_clanht LAPACK_GLOBAL(clanht,CLANHT)
lapack_float_return LAPACK_clanht(
    char const* norm, lapack_int const* n,
    float const* D,
    lapack_complex_float const* E );
#endif

#ifndef LAPACK_zlanht
#define LAPACK_zlanht LAPACK_GLOBAL(zlanht,ZLANHT)
double LAPACK_zlanht(
    char const* norm, lapack_int const* n,
    double const* D,
    lapack_complex_double const* E );
#endif

}  // extern "C"

// --------------------
// wrappers around LAPACK (not in LAPACKE)
inline float LAPACKE_lanst(
    char norm, lapack_int n,
    float* D,
    float* E )
{
    return LAPACK_slanst(
        &norm, &n,
        D,
        E );
}

inline double LAPACKE_lanst(
    char norm, lapack_int n,
    double* D,
    double* E )
{
    return LAPACK_dlanst(
        &norm, &n,
        D,
        E );
}

inline float LAPACKE_lanht(
    char norm, lapack_int n,
    float* D,
    float* E )
{
    return LAPACK_slanst(
        &norm, &n,
        D,
        E );
}

inline double LAPACKE_lanht(
    char norm, lapack_int n,
    double* D,
    double* E )
{
    return LAPACK_dlanst(
        &norm, &n,
        D,
        E );
}

inline float LAPACKE_lanht(
    char norm, lapack_int n,
    float* D,
    std::complex<float>* E )
{
    return LAPACK_clanht(
        &norm, &n,
        D,
        (lapack_complex_float*) E );
}

inline double LAPACKE_lanht(
    char norm, lapack_int n,
    double* D,
    std::complex<double>* E )
{
    return LAPACK_zlanht(
        &norm, &n,
        D,
        (lapack_complex_double*) E );
}

// -----------------------------------------------------------------------------
// Fortran prototypes if not given via lapacke.h
extern "C" {

/* ----- matrix norm - symmetric banded */
#ifndef LAPACK_slansb
#define LAPACK_slansb LAPACK_GLOBAL(slansb,SLANSB)
lapack_float_return LAPACK_slansb(
    char const* norm, char const* uplo,
    lapack_int const* n, lapack_int const* kd,
    float const* AB, lapack_int const* ldab,
    float* work );
#endif

#ifndef LAPACK_dlansb
#define LAPACK_dlansb LAPACK_GLOBAL(dlansb,DLANSB)
double LAPACK_dlansb(
    char const* norm, char const* uplo,
    lapack_int const* n, lapack_int const* kd,
    double const* AB, lapack_int const* ldab,
    double* work );
#endif

#ifndef LAPACK_clansb
#define LAPACK_clansb LAPACK_GLOBAL(clansb,CLANSB)
lapack_float_return LAPACK_clansb(
    char const* norm, char const* uplo,
    lapack_int const* n, lapack_int const* kd,
    lapack_complex_float const* AB, lapack_int const* ldab,
    float* work );
#endif

#ifndef LAPACK_zlansb
#define LAPACK_zlansb LAPACK_GLOBAL(zlansb,ZLANSB)
double LAPACK_zlansb(
    char const* norm, char const* uplo,
    lapack_int const* n, lapack_int const* kd,
    lapack_complex_double const* AB, lapack_int const* ldab,
    double* work );
#endif

}  // extern "C"

// --------------------
// wrappers around LAPACK (not in LAPACKE)
inline float LAPACKE_lansb(
    char norm, char uplo, lapack_int n, lapack_int kd,
    float* AB, lapack_int ldab )
{
    std::vector< float > work( n );
    return LAPACK_slansb(
        &norm, &uplo, &n, &kd,
        AB, &ldab, &work[0] );
}

inline double LAPACKE_lansb(
    char norm, char uplo, lapack_int n, lapack_int kd,
    double* AB, lapack_int ldab )
{
    std::vector< double > work( n );
    return LAPACK_dlansb(
        &norm, &uplo, &n, &kd,
        AB, &ldab, &work[0] );
}

inline float LAPACKE_lansb(
    char norm, char uplo, lapack_int n, lapack_int kd,
    std::complex<float>* AB, lapack_int ldab )
{
    std::vector< float > work( n );
    return LAPACK_clansb(
        &norm, &uplo, &n, &kd,
        (lapack_complex_float*) AB, &ldab, &work[0] );
}

inline double LAPACKE_lansb(
    char norm, char uplo, lapack_int n, lapack_int kd,
    std::complex<double>* AB, lapack_int ldab )
{
    std::vector< double > work( n );
    return LAPACK_zlansb(
        &norm, &uplo, &n, &kd,
        (lapack_complex_double*) AB, &ldab, &work[0] );
}

// -----------------------------------------------------------------------------
// Fortran prototypes if not given via lapacke.h
extern "C" {

/* ----- matrix norm - symmetric packed */
#ifndef LAPACK_slansp
#define LAPACK_slansp LAPACK_GLOBAL(slansp,SLANSP)
lapack_float_return LAPACK_slansp(
    char const* norm, char const* uplo,
    lapack_int const* n,
    float const* AP, float* work );
#endif

#ifndef LAPACK_dlansp
#define LAPACK_dlansp LAPACK_GLOBAL(dlansp,DLANSP)
double LAPACK_dlansp(
    char const* norm, char const* uplo,
    lapack_int const* n,
    double const* AP, double* work );
#endif

#ifndef LAPACK_clansp
#define LAPACK_clansp LAPACK_GLOBAL(clansp,CLANSP)
lapack_float_return LAPACK_clansp(
    char const* norm, char const* uplo,
    lapack_int const* n,
    lapack_complex_float const* AP, float* work );
#endif

#ifndef LAPACK_zlansp
#define LAPACK_zlansp LAPACK_GLOBAL(zlansp,ZLANSP)
double LAPACK_zlansp(
    char const* norm, char const* uplo,
    lapack_int const* n,
    lapack_complex_double const* AP, double* work );
#endif

}  // extern "C"

// --------------------
// wrappers around LAPACK (not in LAPACKE)
inline float LAPACKE_lansp(
    char norm, char uplo, lapack_int n,
    float* AP )
{
    std::vector< float > work( n );
    return LAPACK_slansp(
        &norm, &uplo, &n,
        AP, &work[0] );
}

inline double LAPACKE_lansp(
    char norm, char uplo, lapack_int n,
    double* AP )
{
    std::vector< double > work( n );
    return LAPACK_dlansp(
        &norm, &uplo, &n,
        AP, &work[0] );
}

inline float LAPACKE_lansp(
    char norm, char uplo, lapack_int n,
    std::complex<float>* AP )
{
    std::vector< float > work( n );
    return LAPACK_clansp(
        &norm, &uplo, &n,
        (lapack_complex_float*) AP, &work[0] );
}

inline double LAPACKE_lansp(
    char norm, char uplo, lapack_int n,
    std::complex<double>* AP )
{
    std::vector< double > work( n );
    return LAPACK_zlansp(
        &norm, &uplo, &n,
        (lapack_complex_double*) AP, &work[0] );
}

// -----------------------------------------------------------------------------
inline float LAPACKE_lansy(
    char norm, char uplo, lapack_int n,
    float* A, lapack_int lda )
{
    return LAPACKE_slansy(
        LAPACK_COL_MAJOR, norm, uplo, n,
        A, lda );
}

inline double LAPACKE_lansy(
    char norm, char uplo, lapack_int n,
    double* A, lapack_int lda )
{
    return LAPACKE_dlansy(
        LAPACK_COL_MAJOR, norm, uplo, n,
        A, lda );
}

inline float LAPACKE_lansy(
    char norm, char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda )
{
    return LAPACKE_clansy(
        LAPACK_COL_MAJOR, norm, uplo, n,
        (lapack_complex_float*) A, lda );
}

inline double LAPACKE_lansy(
    char norm, char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda )
{
    return LAPACKE_zlansy(
        LAPACK_COL_MAJOR, norm, uplo, n,
        (lapack_complex_double*) A, lda );
}

// -----------------------------------------------------------------------------
// Fortran prototypes if not given via lapacke.h
extern "C" {

/* ----- matrix norm - triangular banded */
#ifndef LAPACK_slantb
#define LAPACK_slantb LAPACK_GLOBAL(slantb,SLANTB)
lapack_float_return LAPACK_slantb(
    char const* norm, char const* uplo, char const* diag,
    lapack_int const* n, lapack_int const* k,
    float const* AB, lapack_int const* ldab,
    float* work );
#endif

#ifndef LAPACK_dlantb
#define LAPACK_dlantb LAPACK_GLOBAL(dlantb,DLANTB)
double LAPACK_dlantb(
    char const* norm, char const* uplo, char const* diag,
    lapack_int const* n, lapack_int const* k,
    double const* AB, lapack_int const* ldab,
    double* work );
#endif

#ifndef LAPACK_clantb
#define LAPACK_clantb LAPACK_GLOBAL(clantb,CLANTB)
lapack_float_return LAPACK_clantb(
    char const* norm, char const* uplo, char const* diag,
    lapack_int const* n, lapack_int const* k,
    lapack_complex_float const* AB, lapack_int const* ldab,
    float* work );
#endif

#ifndef LAPACK_zlantb
#define LAPACK_zlantb LAPACK_GLOBAL(zlantb,ZLANTB)
double LAPACK_zlantb(
    char const* norm, char const* uplo, char const* diag,
    lapack_int const* n, lapack_int const* k,
    lapack_complex_double const* AB, lapack_int const* ldab,
    double* work );
#endif

}  // extern "C"

// --------------------
// wrappers around LAPACK (not in LAPACKE)
inline float LAPACKE_lantb(
    char norm, char uplo, char diag, lapack_int n, lapack_int k,
    float* AB, lapack_int ldab )
{
    std::vector< float > work( n );
    return LAPACK_slantb(
        &norm, &uplo, &diag, &n, &k,
        AB, &ldab, &work[0] );
}

inline double LAPACKE_lantb(
    char norm, char uplo, char diag, lapack_int n, lapack_int k,
    double* AB, lapack_int ldab )
{
    std::vector< double > work( n );
    return LAPACK_dlantb(
        &norm, &uplo, &diag, &n, &k,
        AB, &ldab, &work[0] );
}

inline float LAPACKE_lantb(
    char norm, char uplo, char diag, lapack_int n, lapack_int k,
    std::complex<float>* AB, lapack_int ldab )
{
    std::vector< float > work( n );
    return LAPACK_clantb(
        &norm, &uplo, &diag, &n, &k,
        (lapack_complex_float*) AB, &ldab, &work[0] );
}

inline double LAPACKE_lantb(
    char norm, char uplo, char diag, lapack_int n, lapack_int k,
    std::complex<double>* AB, lapack_int ldab )
{
    std::vector< double > work( n );
    return LAPACK_zlantb(
        &norm, &uplo, &diag, &n, &k,
        (lapack_complex_double*) AB, &ldab, &work[0] );
}

// -----------------------------------------------------------------------------
// Fortran prototypes if not given via lapacke.h
extern "C" {

/* ----- matrix norm - triangular packed */
#ifndef LAPACK_slantp
#define LAPACK_slantp LAPACK_GLOBAL(slantp,SLANTP)
lapack_float_return LAPACK_slantp(
    char const* norm, char const* uplo, char const* diag,
    lapack_int const* n,
    float const* AP,
    float* work );
#endif

#ifndef LAPACK_dlantp
#define LAPACK_dlantp LAPACK_GLOBAL(dlantp,DLANTP)
double LAPACK_dlantp(
    char const* norm, char const* uplo, char const* diag,
    lapack_int const* n,
    double const* AP,
    double* work );
#endif

#ifndef LAPACK_clantp
#define LAPACK_clantp LAPACK_GLOBAL(clantp,CLANTP)
lapack_float_return LAPACK_clantp(
    char const* norm, char const* uplo, char const* diag,
    lapack_int const* n,
    lapack_complex_float const* AP,
    float* work );
#endif

#ifndef LAPACK_zlantp
#define LAPACK_zlantp LAPACK_GLOBAL(zlantp,ZLANTP)
double LAPACK_zlantp(
    char const* norm, char const* uplo, char const* diag,
    lapack_int const* n,
    lapack_complex_double const* AP,
    double* work );
#endif

}  // extern "C"

// --------------------
// wrappers around LAPACK (not in LAPACKE)
inline float LAPACKE_lantp(
    char norm, char uplo, char diag, lapack_int n,
    float* AP )
{
    std::vector< float > work( n );
    return LAPACK_slantp(
        &norm, &uplo, &diag, &n,
        AP, &work[0] );
}

inline double LAPACKE_lantp(
    char norm, char uplo, char diag, lapack_int n,
    double* AP )
{
    std::vector< double > work( n );
    return LAPACK_dlantp(
        &norm, &uplo, &diag, &n,
        AP, &work[0] );
}

inline float LAPACKE_lantp(
    char norm, char uplo, char diag, lapack_int n,
    std::complex<float>* AP )
{
    std::vector< float > work( n );
    return LAPACK_clantp(
        &norm, &uplo, &diag, &n,
        (lapack_complex_float*) AP, &work[0] );
}

inline double LAPACKE_lantp(
    char norm, char uplo, char diag, lapack_int n,
    std::complex<double>* AP )
{
    std::vector< double > work( n );
    return LAPACK_zlantp(
        &norm, &uplo, &diag, &n,
        (lapack_complex_double*) AP, &work[0] );
}

// -----------------------------------------------------------------------------
inline float LAPACKE_lantr(
    char norm, char uplo, char diag, lapack_int m, lapack_int n,
    float* A, lapack_int lda )
{
    return LAPACKE_slantr(
        LAPACK_COL_MAJOR, norm, uplo, diag, m, n,
        A, lda );
}

inline double LAPACKE_lantr(
    char norm, char uplo, char diag, lapack_int m, lapack_int n,
    double* A, lapack_int lda )
{
    return LAPACKE_dlantr(
        LAPACK_COL_MAJOR, norm, uplo, diag, m, n,
        A, lda );
}

inline float LAPACKE_lantr(
    char norm, char uplo, char diag, lapack_int m, lapack_int n,
    std::complex<float>* A, lapack_int lda )
{
    return LAPACKE_clantr(
        LAPACK_COL_MAJOR, norm, uplo, diag, m, n,
        (lapack_complex_float*) A, lda );
}

inline double LAPACKE_lantr(
    char norm, char uplo, char diag, lapack_int m, lapack_int n,
    std::complex<double>* A, lapack_int lda )
{
    return LAPACKE_zlantr(
        LAPACK_COL_MAJOR, norm, uplo, diag, m, n,
        (lapack_complex_double*) A, lda );
}


// -----------------------------------------------------------------------------
// Fortran prototypes if not given via lapacke.h
extern "C" {

/* ----- apply Householder reflector */
#ifndef LAPACK_slarf
#define LAPACK_slarf LAPACK_GLOBAL(slarf,SLARF)
void LAPACK_slarf(
    char const* side,
    lapack_int const* m, lapack_int const* n,
    float const* v, lapack_int const* incv,
    float const* tau,
    float* c, lapack_int const* ldc,
    float* work );
#endif

#ifndef LAPACK_dlarf
#define LAPACK_dlarf LAPACK_GLOBAL(dlarf,DLARF)
void LAPACK_dlarf(
    char const* side,
    lapack_int const* m, lapack_int const* n,
    double const* v, lapack_int const* incv,
    double const* tau,
    double* c, lapack_int const* ldc,
    double* work );
#endif

#ifndef LAPACK_clarf
#define LAPACK_clarf LAPACK_GLOBAL(clarf,CLARF)
void LAPACK_clarf(
    char const* side,
    lapack_int const* m, lapack_int const* n,
    lapack_complex_float const* v, lapack_int const* incv,
    lapack_complex_float const* tau,
    lapack_complex_float* c, lapack_int const* ldc,
    lapack_complex_float* work );
#endif

#ifndef LAPACK_zlarf
#define LAPACK_zlarf LAPACK_GLOBAL(zlarf,ZLARF)
void LAPACK_zlarf(
    char const* side,
    lapack_int const* m, lapack_int const* n,
    lapack_complex_double const* v, lapack_int const* incv,
    lapack_complex_double const* tau,
    lapack_complex_double* c, lapack_int const* ldc,
    lapack_complex_double* work );
#endif

}  // extern "C"

// --------------------
// wrappers around LAPACK (not in LAPACKE)
// todo: LAPACK has no error checks for larf
inline lapack_int LAPACKE_larf(
    char side, lapack_int m, lapack_int n,
    float* V, lapack_int incv, float tau,
    float* C, lapack_int ldc )
{
    std::vector< float > work( side == 'l' || side == 'L' ? n : m );
    LAPACK_slarf(
        &side, &m, &n,
        V, &incv,
        &tau,
        C, &ldc, &work[0] );
    return 0;
}

inline lapack_int LAPACKE_larf(
    char side, lapack_int m, lapack_int n,
    double* V, lapack_int incv, double tau,
    double* C, lapack_int ldc )
{
    std::vector< double > work( side == 'l' || side == 'L' ? n : m );
    LAPACK_dlarf(
        &side, &m, &n,
        V, &incv,
        &tau,
        C, &ldc, &work[0] );
    return 0;
}

inline lapack_int LAPACKE_larf(
    char side, lapack_int m, lapack_int n,
    std::complex<float>* V, lapack_int incv, std::complex<float> tau,
    std::complex<float>* C, lapack_int ldc )
{
    std::vector< lapack_complex_float > work( side == 'l' || side == 'L' ? n : m );
    LAPACK_clarf(
        &side, &m, &n,
        (lapack_complex_float*) V, &incv,
        (lapack_complex_float*) &tau,
        (lapack_complex_float*) C, &ldc, &work[0] );
    return 0;
}

inline lapack_int LAPACKE_larf(
    char side, lapack_int m, lapack_int n,
    std::complex<double>* V, lapack_int incv, std::complex<double> tau,
    std::complex<double>* C, lapack_int ldc )
{
    std::vector< lapack_complex_double > work( side == 'l' || side == 'L' ? n : m );
    LAPACK_zlarf(
        &side, &m, &n,
        (lapack_complex_double*) V, &incv,
        (lapack_complex_double*) &tau,
        (lapack_complex_double*) C, &ldc, &work[0] );
    return 0;
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_larfb(
    char side, char trans, char direction, char storev, lapack_int m, lapack_int n, lapack_int k,
    float* V, lapack_int ldv,
    float* T, lapack_int ldt,
    float* C, lapack_int ldc )
{
    return LAPACKE_slarfb(
        LAPACK_COL_MAJOR, side, trans, direction, storev, m, n, k,
        V, ldv,
        T, ldt,
        C, ldc );
}

inline lapack_int LAPACKE_larfb(
    char side, char trans, char direction, char storev, lapack_int m, lapack_int n, lapack_int k,
    double* V, lapack_int ldv,
    double* T, lapack_int ldt,
    double* C, lapack_int ldc )
{
    return LAPACKE_dlarfb(
        LAPACK_COL_MAJOR, side, trans, direction, storev, m, n, k,
        V, ldv,
        T, ldt,
        C, ldc );
}

inline lapack_int LAPACKE_larfb(
    char side, char trans, char direction, char storev, lapack_int m, lapack_int n, lapack_int k,
    std::complex<float>* V, lapack_int ldv,
    std::complex<float>* T, lapack_int ldt,
    std::complex<float>* C, lapack_int ldc )
{
    return LAPACKE_clarfb(
        LAPACK_COL_MAJOR, side, trans, direction, storev, m, n, k,
        (lapack_complex_float*) V, ldv,
        (lapack_complex_float*) T, ldt,
        (lapack_complex_float*) C, ldc );
}

inline lapack_int LAPACKE_larfb(
    char side, char trans, char direction, char storev, lapack_int m, lapack_int n, lapack_int k,
    std::complex<double>* V, lapack_int ldv,
    std::complex<double>* T, lapack_int ldt,
    std::complex<double>* C, lapack_int ldc )
{
    return LAPACKE_zlarfb(
        LAPACK_COL_MAJOR, side, trans, direction, storev, m, n, k,
        (lapack_complex_double*) V, ldv,
        (lapack_complex_double*) T, ldt,
        (lapack_complex_double*) C, ldc );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_larfg(
    lapack_int n,
    float* alpha,
    float* X, lapack_int incx,
    float* tau )
{
    return LAPACKE_slarfg(
        n,
        alpha,
        X, incx,
        tau );
}

inline lapack_int LAPACKE_larfg(
    lapack_int n,
    double* alpha,
    double* X, lapack_int incx,
    double* tau )
{
    return LAPACKE_dlarfg(
        n,
        alpha,
        X, incx,
        tau );
}

inline lapack_int LAPACKE_larfg(
    lapack_int n,
    std::complex<float>* alpha,
    std::complex<float>* X, lapack_int incx,
    std::complex<float>* tau )
{
    return LAPACKE_clarfg(
        n,
        (lapack_complex_float*) alpha,
        (lapack_complex_float*) X, incx,
        (lapack_complex_float*) tau );
}

inline lapack_int LAPACKE_larfg(
    lapack_int n,
    std::complex<double>* alpha,
    std::complex<double>* X, lapack_int incx,
    std::complex<double>* tau )
{
    return LAPACKE_zlarfg(
        n,
        (lapack_complex_double*) alpha,
        (lapack_complex_double*) X, incx,
        (lapack_complex_double*) tau );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_larft(
    char direction, char storev, lapack_int n, lapack_int k,
    float* V, lapack_int ldv,
    float* tau,
    float* T, lapack_int ldt )
{
    return LAPACKE_slarft(
        LAPACK_COL_MAJOR, direction, storev, n, k,
        V, ldv,
        tau,
        T, ldt );
}

inline lapack_int LAPACKE_larft(
    char direction, char storev, lapack_int n, lapack_int k,
    double* V, lapack_int ldv,
    double* tau,
    double* T, lapack_int ldt )
{
    return LAPACKE_dlarft(
        LAPACK_COL_MAJOR, direction, storev, n, k,
        V, ldv,
        tau,
        T, ldt );
}

inline lapack_int LAPACKE_larft(
    char direction, char storev, lapack_int n, lapack_int k,
    std::complex<float>* V, lapack_int ldv,
    std::complex<float>* tau,
    std::complex<float>* T, lapack_int ldt )
{
    return LAPACKE_clarft(
        LAPACK_COL_MAJOR, direction, storev, n, k,
        (lapack_complex_float*) V, ldv,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) T, ldt );
}

inline lapack_int LAPACKE_larft(
    char direction, char storev, lapack_int n, lapack_int k,
    std::complex<double>* V, lapack_int ldv,
    std::complex<double>* tau,
    std::complex<double>* T, lapack_int ldt )
{
    return LAPACKE_zlarft(
        LAPACK_COL_MAJOR, direction, storev, n, k,
        (lapack_complex_double*) V, ldv,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) T, ldt );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_larfx(
    char side, lapack_int m, lapack_int n,
    float* V, float tau,
    float* C, lapack_int ldc )
{
    std::vector< float > work( side == 'l' || side == 'L' ? n : m );
    return LAPACKE_slarfx(
        LAPACK_COL_MAJOR, side, m, n,
        V, tau,
        C, ldc, &work[0] );
}

inline lapack_int LAPACKE_larfx(
    char side, lapack_int m, lapack_int n,
    double* V, double tau,
    double* C, lapack_int ldc )
{
    std::vector< double > work( side == 'l' || side == 'L' ? n : m );
    return LAPACKE_dlarfx(
        LAPACK_COL_MAJOR, side, m, n,
        V, tau,
        C, ldc, &work[0] );
}

inline lapack_int LAPACKE_larfx(
    char side, lapack_int m, lapack_int n,
    std::complex<float>* V, std::complex<float> tau,
    std::complex<float>* C, lapack_int ldc )
{
    std::vector< lapack_complex_float > work( side == 'l' || side == 'L' ? n : m );
    return LAPACKE_clarfx(
        LAPACK_COL_MAJOR, side, m, n,
        (lapack_complex_float*) V,
      *((lapack_complex_float*) &tau),
        (lapack_complex_float*) C, ldc, &work[0] );
}

inline lapack_int LAPACKE_larfx(
    char side, lapack_int m, lapack_int n,
    std::complex<double>* V, std::complex<double> tau,
    std::complex<double>* C, lapack_int ldc )
{
    std::vector< lapack_complex_double > work( side == 'l' || side == 'L' ? n : m );
    return LAPACKE_zlarfx(
        LAPACK_COL_MAJOR, side, m, n,
        (lapack_complex_double*) V,
      *((lapack_complex_double*) &tau),
        (lapack_complex_double*) C, ldc, &work[0] );
}

// -----------------------------------------------------------------------------
#if LAPACK_VERSION >= 30700  // >= 3.7.0

// Fortran prototypes not given via lapacke.h
extern "C" {

#ifndef LAPACK_slarfy
#define LAPACK_slarfy LAPACK_GLOBAL(slarfy,SLARFY)
void LAPACK_slarfy(
    char const* uplo,
    lapack_int const* n,
    float const* V, lapack_int const* incv,
    float const* tau,
    float* C, lapack_int const* ldc,
    float* work );
#endif

#ifndef LAPACK_dlarfy
#define LAPACK_dlarfy LAPACK_GLOBAL(dlarfy,DLARFY)
void LAPACK_dlarfy(
    char const* uplo,
    lapack_int const* n,
    double const* V, lapack_int const* incv,
    double const* tau,
    double* C, lapack_int const* ldc,
    double* work );
#endif

#ifndef LAPACK_clarfy
#define LAPACK_clarfy LAPACK_GLOBAL(clarfy,CLARFY)
void LAPACK_clarfy(
    char const* uplo,
    lapack_int const* n,
    lapack_complex_float const* V, lapack_int const* incv,
    lapack_complex_float const* tau,
    lapack_complex_float* C, lapack_int const* ldc,
    lapack_complex_float* work );
#endif

#ifndef LAPACK_zlarfy
#define LAPACK_zlarfy LAPACK_GLOBAL(zlarfy,ZLARFY)
void LAPACK_zlarfy(
    char const* uplo,
    lapack_int const* n,
    lapack_complex_double const* V, lapack_int const* incv,
    lapack_complex_double const* tau,
    lapack_complex_double* C, lapack_int const* ldc,
    lapack_complex_double* work );
#endif

}  // extern "C"

// --------------------
// wrappers around LAPACK (not in LAPACKE)
inline lapack_int LAPACKE_larfy(
    char uplo, lapack_int n,
    float* v, lapack_int incv, float tau,
    float* C, lapack_int ldc )
{
    std::vector<float> work( n );
    LAPACK_slarfy(
        &uplo, &n,
        v, &incv, &tau,
        C, &ldc, &work[0] );
    return 0;
}

inline lapack_int LAPACKE_larfy(
    char uplo, lapack_int n,
    double* v, lapack_int incv, double tau,
    double* C, lapack_int ldc )
{
    std::vector<double> work( n );
    LAPACK_dlarfy(
        &uplo, &n,
        v, &incv, &tau,
        C, &ldc, &work[0] );
    return 0;
}

inline lapack_int LAPACKE_larfy(
    char uplo, lapack_int n,
    std::complex<float>* v, lapack_int incv, std::complex<float> tau,
    std::complex<float>* C, lapack_int ldc )
{
    std::vector<lapack_complex_float> work( n );
    LAPACK_clarfy(
        &uplo, &n,
        (lapack_complex_float*) v, &incv, (lapack_complex_float*) &tau,
        (lapack_complex_float*) C, &ldc, &work[0] );
    return 0;
}

inline lapack_int LAPACKE_larfy(
    char uplo, lapack_int n,
    std::complex<double>* v, lapack_int incv, std::complex<double> tau,
    std::complex<double>* C, lapack_int ldc )
{
    std::vector<lapack_complex_double> work( n );
    LAPACK_zlarfy(
        &uplo, &n,
        (lapack_complex_double*) v, &incv, (lapack_complex_double*) &tau,
        (lapack_complex_double*) C, &ldc, &work[0] );
    return 0;
}
#endif // 30700

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_laset(
    char uplo, lapack_int m, lapack_int n, float alpha, float beta,
    float* A, lapack_int lda )
{
    return LAPACKE_slaset(
        LAPACK_COL_MAJOR, uplo, m, n, alpha, beta,
        A, lda );
}

inline lapack_int LAPACKE_laset(
    char uplo, lapack_int m, lapack_int n, double alpha, double beta,
    double* A, lapack_int lda )
{
    return LAPACKE_dlaset(
        LAPACK_COL_MAJOR, uplo, m, n, alpha, beta,
        A, lda );
}

inline lapack_int LAPACKE_laset(
    char uplo, lapack_int m, lapack_int n, std::complex<float> alpha, std::complex<float> beta,
    std::complex<float>* A, lapack_int lda )
{
    return LAPACKE_claset(
        LAPACK_COL_MAJOR, uplo, m, n,
      *((lapack_complex_float*) &alpha),
      *((lapack_complex_float*) &beta),
        (lapack_complex_float*) A, lda );
}

inline lapack_int LAPACKE_laset(
    char uplo, lapack_int m, lapack_int n, std::complex<double> alpha, std::complex<double> beta,
    std::complex<double>* A, lapack_int lda )
{
    return LAPACKE_zlaset(
        LAPACK_COL_MAJOR, uplo, m, n,
      *((lapack_complex_double*) &alpha),
      *((lapack_complex_double*) &beta),
        (lapack_complex_double*) A, lda );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_laswp(
    lapack_int n,
    float* A, lapack_int lda, lapack_int k1, lapack_int k2,
    lapack_int* ipiv, lapack_int incx )
{
    return LAPACKE_slaswp(
        LAPACK_COL_MAJOR, n,
        A, lda, k1, k2,
        ipiv, incx );
}

inline lapack_int LAPACKE_laswp(
    lapack_int n,
    double* A, lapack_int lda, lapack_int k1, lapack_int k2,
    lapack_int* ipiv, lapack_int incx )
{
    return LAPACKE_dlaswp(
        LAPACK_COL_MAJOR, n,
        A, lda, k1, k2,
        ipiv, incx );
}

inline lapack_int LAPACKE_laswp(
    lapack_int n,
    std::complex<float>* A, lapack_int lda, lapack_int k1, lapack_int k2,
    lapack_int* ipiv, lapack_int incx )
{
    return LAPACKE_claswp(
        LAPACK_COL_MAJOR, n,
        (lapack_complex_float*) A, lda, k1, k2,
        ipiv, incx );
}

inline lapack_int LAPACKE_laswp(
    lapack_int n,
    std::complex<double>* A, lapack_int lda, lapack_int k1, lapack_int k2,
    lapack_int* ipiv, lapack_int incx )
{
    return LAPACKE_zlaswp(
        LAPACK_COL_MAJOR, n,
        (lapack_complex_double*) A, lda, k1, k2,
        ipiv, incx );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_pbcon(
    char uplo, lapack_int n, lapack_int kd,
    float* AB, lapack_int ldab, float anorm,
    float* rcond )
{
    return LAPACKE_spbcon(
        LAPACK_COL_MAJOR, uplo, n, kd,
        AB, ldab, anorm,
        rcond );
}

inline lapack_int LAPACKE_pbcon(
    char uplo, lapack_int n, lapack_int kd,
    double* AB, lapack_int ldab, double anorm,
    double* rcond )
{
    return LAPACKE_dpbcon(
        LAPACK_COL_MAJOR, uplo, n, kd,
        AB, ldab, anorm,
        rcond );
}

inline lapack_int LAPACKE_pbcon(
    char uplo, lapack_int n, lapack_int kd,
    std::complex<float>* AB, lapack_int ldab, float anorm,
    float* rcond )
{
    return LAPACKE_cpbcon(
        LAPACK_COL_MAJOR, uplo, n, kd,
        (lapack_complex_float*) AB, ldab, anorm,
        rcond );
}

inline lapack_int LAPACKE_pbcon(
    char uplo, lapack_int n, lapack_int kd,
    std::complex<double>* AB, lapack_int ldab, double anorm,
    double* rcond )
{
    return LAPACKE_zpbcon(
        LAPACK_COL_MAJOR, uplo, n, kd,
        (lapack_complex_double*) AB, ldab, anorm,
        rcond );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_pbequ(
    char uplo, lapack_int n, lapack_int kd,
    float* AB, lapack_int ldab,
    float* S,
    float* scond,
    float* amax )
{
    return LAPACKE_spbequ(
        LAPACK_COL_MAJOR, uplo, n, kd,
        AB, ldab,
        S,
        scond,
        amax );
}

inline lapack_int LAPACKE_pbequ(
    char uplo, lapack_int n, lapack_int kd,
    double* AB, lapack_int ldab,
    double* S,
    double* scond,
    double* amax )
{
    return LAPACKE_dpbequ(
        LAPACK_COL_MAJOR, uplo, n, kd,
        AB, ldab,
        S,
        scond,
        amax );
}

inline lapack_int LAPACKE_pbequ(
    char uplo, lapack_int n, lapack_int kd,
    std::complex<float>* AB, lapack_int ldab,
    float* S,
    float* scond,
    float* amax )
{
    return LAPACKE_cpbequ(
        LAPACK_COL_MAJOR, uplo, n, kd,
        (lapack_complex_float*) AB, ldab,
        S,
        scond,
        amax );
}

inline lapack_int LAPACKE_pbequ(
    char uplo, lapack_int n, lapack_int kd,
    std::complex<double>* AB, lapack_int ldab,
    double* S,
    double* scond,
    double* amax )
{
    return LAPACKE_zpbequ(
        LAPACK_COL_MAJOR, uplo, n, kd,
        (lapack_complex_double*) AB, ldab,
        S,
        scond,
        amax );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_pbrfs(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs,
    float* AB, lapack_int ldab,
    float* AFB, lapack_int ldafb,
    float* B, lapack_int ldb,
    float* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_spbrfs(
        LAPACK_COL_MAJOR, uplo, n, kd, nrhs,
        AB, ldab,
        AFB, ldafb,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_pbrfs(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs,
    double* AB, lapack_int ldab,
    double* AFB, lapack_int ldafb,
    double* B, lapack_int ldb,
    double* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_dpbrfs(
        LAPACK_COL_MAJOR, uplo, n, kd, nrhs,
        AB, ldab,
        AFB, ldafb,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_pbrfs(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs,
    std::complex<float>* AB, lapack_int ldab,
    std::complex<float>* AFB, lapack_int ldafb,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_cpbrfs(
        LAPACK_COL_MAJOR, uplo, n, kd, nrhs,
        (lapack_complex_float*) AB, ldab,
        (lapack_complex_float*) AFB, ldafb,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_pbrfs(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs,
    std::complex<double>* AB, lapack_int ldab,
    std::complex<double>* AFB, lapack_int ldafb,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_zpbrfs(
        LAPACK_COL_MAJOR, uplo, n, kd, nrhs,
        (lapack_complex_double*) AB, ldab,
        (lapack_complex_double*) AFB, ldafb,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) X, ldx,
        ferr,
        berr );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_pbsv(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs,
    float* AB, lapack_int ldab,
    float* B, lapack_int ldb )
{
    return LAPACKE_spbsv(
        LAPACK_COL_MAJOR, uplo, n, kd, nrhs,
        AB, ldab,
        B, ldb );
}

inline lapack_int LAPACKE_pbsv(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs,
    double* AB, lapack_int ldab,
    double* B, lapack_int ldb )
{
    return LAPACKE_dpbsv(
        LAPACK_COL_MAJOR, uplo, n, kd, nrhs,
        AB, ldab,
        B, ldb );
}

inline lapack_int LAPACKE_pbsv(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs,
    std::complex<float>* AB, lapack_int ldab,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cpbsv(
        LAPACK_COL_MAJOR, uplo, n, kd, nrhs,
        (lapack_complex_float*) AB, ldab,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_pbsv(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs,
    std::complex<double>* AB, lapack_int ldab,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zpbsv(
        LAPACK_COL_MAJOR, uplo, n, kd, nrhs,
        (lapack_complex_double*) AB, ldab,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_pbtrf(
    char uplo, lapack_int n, lapack_int kd,
    float* AB, lapack_int ldab )
{
    return LAPACKE_spbtrf(
        LAPACK_COL_MAJOR, uplo, n, kd,
        AB, ldab );
}

inline lapack_int LAPACKE_pbtrf(
    char uplo, lapack_int n, lapack_int kd,
    double* AB, lapack_int ldab )
{
    return LAPACKE_dpbtrf(
        LAPACK_COL_MAJOR, uplo, n, kd,
        AB, ldab );
}

inline lapack_int LAPACKE_pbtrf(
    char uplo, lapack_int n, lapack_int kd,
    std::complex<float>* AB, lapack_int ldab )
{
    return LAPACKE_cpbtrf(
        LAPACK_COL_MAJOR, uplo, n, kd,
        (lapack_complex_float*) AB, ldab );
}

inline lapack_int LAPACKE_pbtrf(
    char uplo, lapack_int n, lapack_int kd,
    std::complex<double>* AB, lapack_int ldab )
{
    return LAPACKE_zpbtrf(
        LAPACK_COL_MAJOR, uplo, n, kd,
        (lapack_complex_double*) AB, ldab );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_pbtrs(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs,
    float* AB, lapack_int ldab,
    float* B, lapack_int ldb )
{
    return LAPACKE_spbtrs(
        LAPACK_COL_MAJOR, uplo, n, kd, nrhs,
        AB, ldab,
        B, ldb );
}

inline lapack_int LAPACKE_pbtrs(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs,
    double* AB, lapack_int ldab,
    double* B, lapack_int ldb )
{
    return LAPACKE_dpbtrs(
        LAPACK_COL_MAJOR, uplo, n, kd, nrhs,
        AB, ldab,
        B, ldb );
}

inline lapack_int LAPACKE_pbtrs(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs,
    std::complex<float>* AB, lapack_int ldab,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cpbtrs(
        LAPACK_COL_MAJOR, uplo, n, kd, nrhs,
        (lapack_complex_float*) AB, ldab,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_pbtrs(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs,
    std::complex<double>* AB, lapack_int ldab,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zpbtrs(
        LAPACK_COL_MAJOR, uplo, n, kd, nrhs,
        (lapack_complex_double*) AB, ldab,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_pocon(
    char uplo, lapack_int n,
    float* A, lapack_int lda, float anorm,
    float* rcond )
{
    return LAPACKE_spocon(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda, anorm,
        rcond );
}

inline lapack_int LAPACKE_pocon(
    char uplo, lapack_int n,
    double* A, lapack_int lda, double anorm,
    double* rcond )
{
    return LAPACKE_dpocon(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda, anorm,
        rcond );
}

inline lapack_int LAPACKE_pocon(
    char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda, float anorm,
    float* rcond )
{
    return LAPACKE_cpocon(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) A, lda, anorm,
        rcond );
}

inline lapack_int LAPACKE_pocon(
    char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda, double anorm,
    double* rcond )
{
    return LAPACKE_zpocon(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) A, lda, anorm,
        rcond );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_poequ(
    lapack_int n,
    float* A, lapack_int lda,
    float* S,
    float* scond,
    float* amax )
{
    return LAPACKE_spoequ(
        LAPACK_COL_MAJOR, n,
        A, lda,
        S,
        scond,
        amax );
}

inline lapack_int LAPACKE_poequ(
    lapack_int n,
    double* A, lapack_int lda,
    double* S,
    double* scond,
    double* amax )
{
    return LAPACKE_dpoequ(
        LAPACK_COL_MAJOR, n,
        A, lda,
        S,
        scond,
        amax );
}

inline lapack_int LAPACKE_poequ(
    lapack_int n,
    std::complex<float>* A, lapack_int lda,
    float* S,
    float* scond,
    float* amax )
{
    return LAPACKE_cpoequ(
        LAPACK_COL_MAJOR, n,
        (lapack_complex_float*) A, lda,
        S,
        scond,
        amax );
}

inline lapack_int LAPACKE_poequ(
    lapack_int n,
    std::complex<double>* A, lapack_int lda,
    double* S,
    double* scond,
    double* amax )
{
    return LAPACKE_zpoequ(
        LAPACK_COL_MAJOR, n,
        (lapack_complex_double*) A, lda,
        S,
        scond,
        amax );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_porfs(
    char uplo, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    float* AF, lapack_int ldaf,
    float* B, lapack_int ldb,
    float* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_sporfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        AF, ldaf,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_porfs(
    char uplo, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    double* AF, lapack_int ldaf,
    double* B, lapack_int ldb,
    double* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_dporfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        AF, ldaf,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_porfs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* AF, lapack_int ldaf,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_cporfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) AF, ldaf,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_porfs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* AF, lapack_int ldaf,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_zporfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) AF, ldaf,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) X, ldx,
        ferr,
        berr );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_posv(
    char uplo, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    float* B, lapack_int ldb )
{
    return LAPACKE_sposv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        B, ldb );
}

inline lapack_int LAPACKE_posv(
    char uplo, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    double* B, lapack_int ldb )
{
    return LAPACKE_dposv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        B, ldb );
}

inline lapack_int LAPACKE_posv(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cposv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_posv(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zposv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_potrf(
    char uplo, lapack_int n,
    float* A, lapack_int lda )
{
    return LAPACKE_spotrf(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda );
}

inline lapack_int LAPACKE_potrf(
    char uplo, lapack_int n,
    double* A, lapack_int lda )
{
    return LAPACKE_dpotrf(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda );
}

inline lapack_int LAPACKE_potrf(
    char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda )
{
    return LAPACKE_cpotrf(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) A, lda );
}

inline lapack_int LAPACKE_potrf(
    char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda )
{
    return LAPACKE_zpotrf(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) A, lda );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_potri(
    char uplo, lapack_int n,
    float* A, lapack_int lda )
{
    return LAPACKE_spotri(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda );
}

inline lapack_int LAPACKE_potri(
    char uplo, lapack_int n,
    double* A, lapack_int lda )
{
    return LAPACKE_dpotri(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda );
}

inline lapack_int LAPACKE_potri(
    char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda )
{
    return LAPACKE_cpotri(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) A, lda );
}

inline lapack_int LAPACKE_potri(
    char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda )
{
    return LAPACKE_zpotri(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) A, lda );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_potrs(
    char uplo, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    float* B, lapack_int ldb )
{
    return LAPACKE_spotrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        B, ldb );
}

inline lapack_int LAPACKE_potrs(
    char uplo, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    double* B, lapack_int ldb )
{
    return LAPACKE_dpotrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        B, ldb );
}

inline lapack_int LAPACKE_potrs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cpotrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_potrs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zpotrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_ppcon(
    char uplo, lapack_int n,
    float* AP, float anorm,
    float* rcond )
{
    return LAPACKE_sppcon(
        LAPACK_COL_MAJOR, uplo, n,
        AP, anorm,
        rcond );
}

inline lapack_int LAPACKE_ppcon(
    char uplo, lapack_int n,
    double* AP, double anorm,
    double* rcond )
{
    return LAPACKE_dppcon(
        LAPACK_COL_MAJOR, uplo, n,
        AP, anorm,
        rcond );
}

inline lapack_int LAPACKE_ppcon(
    char uplo, lapack_int n,
    std::complex<float>* AP, float anorm,
    float* rcond )
{
    return LAPACKE_cppcon(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) AP, anorm,
        rcond );
}

inline lapack_int LAPACKE_ppcon(
    char uplo, lapack_int n,
    std::complex<double>* AP, double anorm,
    double* rcond )
{
    return LAPACKE_zppcon(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) AP, anorm,
        rcond );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_ppequ(
    char uplo, lapack_int n,
    float* AP,
    float* S,
    float* scond,
    float* amax )
{
    return LAPACKE_sppequ(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        S,
        scond,
        amax );
}

inline lapack_int LAPACKE_ppequ(
    char uplo, lapack_int n,
    double* AP,
    double* S,
    double* scond,
    double* amax )
{
    return LAPACKE_dppequ(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        S,
        scond,
        amax );
}

inline lapack_int LAPACKE_ppequ(
    char uplo, lapack_int n,
    std::complex<float>* AP,
    float* S,
    float* scond,
    float* amax )
{
    return LAPACKE_cppequ(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) AP,
        S,
        scond,
        amax );
}

inline lapack_int LAPACKE_ppequ(
    char uplo, lapack_int n,
    std::complex<double>* AP,
    double* S,
    double* scond,
    double* amax )
{
    return LAPACKE_zppequ(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) AP,
        S,
        scond,
        amax );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_pprfs(
    char uplo, lapack_int n, lapack_int nrhs,
    float* AP,
    float* AFP,
    float* B, lapack_int ldb,
    float* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_spprfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        AFP,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_pprfs(
    char uplo, lapack_int n, lapack_int nrhs,
    double* AP,
    double* AFP,
    double* B, lapack_int ldb,
    double* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_dpprfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        AFP,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_pprfs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* AP,
    std::complex<float>* AFP,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_cpprfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) AFP,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_pprfs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* AP,
    std::complex<double>* AFP,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_zpprfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) AFP,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) X, ldx,
        ferr,
        berr );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_ppsv(
    char uplo, lapack_int n, lapack_int nrhs,
    float* AP,
    float* B, lapack_int ldb )
{
    return LAPACKE_sppsv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        B, ldb );
}

inline lapack_int LAPACKE_ppsv(
    char uplo, lapack_int n, lapack_int nrhs,
    double* AP,
    double* B, lapack_int ldb )
{
    return LAPACKE_dppsv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        B, ldb );
}

inline lapack_int LAPACKE_ppsv(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* AP,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cppsv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_ppsv(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* AP,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zppsv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_pptrf(
    char uplo, lapack_int n,
    float* AP )
{
    return LAPACKE_spptrf(
        LAPACK_COL_MAJOR, uplo, n,
        AP );
}

inline lapack_int LAPACKE_pptrf(
    char uplo, lapack_int n,
    double* AP )
{
    return LAPACKE_dpptrf(
        LAPACK_COL_MAJOR, uplo, n,
        AP );
}

inline lapack_int LAPACKE_pptrf(
    char uplo, lapack_int n,
    std::complex<float>* AP )
{
    return LAPACKE_cpptrf(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) AP );
}

inline lapack_int LAPACKE_pptrf(
    char uplo, lapack_int n,
    std::complex<double>* AP )
{
    return LAPACKE_zpptrf(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) AP );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_pptri(
    char uplo, lapack_int n,
    float* AP )
{
    return LAPACKE_spptri(
        LAPACK_COL_MAJOR, uplo, n,
        AP );
}

inline lapack_int LAPACKE_pptri(
    char uplo, lapack_int n,
    double* AP )
{
    return LAPACKE_dpptri(
        LAPACK_COL_MAJOR, uplo, n,
        AP );
}

inline lapack_int LAPACKE_pptri(
    char uplo, lapack_int n,
    std::complex<float>* AP )
{
    return LAPACKE_cpptri(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) AP );
}

inline lapack_int LAPACKE_pptri(
    char uplo, lapack_int n,
    std::complex<double>* AP )
{
    return LAPACKE_zpptri(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) AP );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_pptrs(
    char uplo, lapack_int n, lapack_int nrhs,
    float* AP,
    float* B, lapack_int ldb )
{
    return LAPACKE_spptrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        B, ldb );
}

inline lapack_int LAPACKE_pptrs(
    char uplo, lapack_int n, lapack_int nrhs,
    double* AP,
    double* B, lapack_int ldb )
{
    return LAPACKE_dpptrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        B, ldb );
}

inline lapack_int LAPACKE_pptrs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* AP,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cpptrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_pptrs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* AP,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zpptrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_ptcon(
    lapack_int n,
    float* D,
    float* E,
    float anorm, float* rcond )
{
    return LAPACKE_sptcon(
        n,
        D,
        E,
        anorm, rcond );
}

inline lapack_int LAPACKE_ptcon(
    lapack_int n,
    double* D,
    double* E,
    double anorm, double* rcond )
{
    return LAPACKE_dptcon(
        n,
        D,
        E,
        anorm, rcond );
}

inline lapack_int LAPACKE_ptcon(
    lapack_int n,
    float* D,
    std::complex<float>* E,
    float anorm, float* rcond )
{
    return LAPACKE_cptcon(
        n,
        D,
        (lapack_complex_float*) E,
        anorm, rcond );
}

inline lapack_int LAPACKE_ptcon(
    lapack_int n,
    double* D,
    std::complex<double>* E,
    double anorm, double* rcond )
{
    return LAPACKE_zptcon(
        n,
        D,
        (lapack_complex_double*) E,
        anorm, rcond );
}

// -----------------------------------------------------------------------------
// for [sd]ptrfs, uplo is ignored
inline lapack_int LAPACKE_ptrfs(
    char uplo, lapack_int n, lapack_int nrhs,
    float* D,
    float* E,
    float* DF,
    float* EF,
    float* B, lapack_int ldb,
    float* X, lapack_int ldx,
    float* ferr, float* berr )
{
    return LAPACKE_sptrfs(
        LAPACK_COL_MAJOR, n, nrhs,
        D,
        E,
        DF,
        EF,
        B, ldb,
        X, ldx,
        ferr, berr );
}

inline lapack_int LAPACKE_ptrfs(
    char uplo, lapack_int n, lapack_int nrhs,
    double* D,
    double* E,
    double* DF,
    double* EF,
    double* B, lapack_int ldb,
    double* X, lapack_int ldx,
    double* ferr, double* berr )
{
    return LAPACKE_dptrfs(
        LAPACK_COL_MAJOR, n, nrhs,
        D,
        E,
        DF,
        EF,
        B, ldb,
        X, ldx,
        ferr, berr );
}

inline lapack_int LAPACKE_ptrfs(
    char uplo, lapack_int n, lapack_int nrhs,
    float* D,
    std::complex<float>* E,
    float* DF,
    std::complex<float>* EF,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* X, lapack_int ldx,
    float* ferr, float* berr )
{
    return LAPACKE_cptrfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        D,
        (lapack_complex_float*) E,
        DF,
        (lapack_complex_float*) EF,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) X, ldx,
        ferr, berr );
}

inline lapack_int LAPACKE_ptrfs(
    char uplo, lapack_int n, lapack_int nrhs,
    double* D,
    std::complex<double>* E,
    double* DF,
    std::complex<double>* EF,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* X, lapack_int ldx,
    double* ferr, double* berr )
{
    return LAPACKE_zptrfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        D,
        (lapack_complex_double*) E,
        DF,
        (lapack_complex_double*) EF,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) X, ldx,
        ferr, berr );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_ptsv(
    lapack_int n, lapack_int nrhs,
    float* D,
    float* E,
    float* B, lapack_int ldb )
{
    return LAPACKE_sptsv(
        LAPACK_COL_MAJOR, n, nrhs,
        D,
        E,
        B, ldb );
}

inline lapack_int LAPACKE_ptsv(
    lapack_int n, lapack_int nrhs,
    double* D,
    double* E,
    double* B, lapack_int ldb )
{
    return LAPACKE_dptsv(
        LAPACK_COL_MAJOR, n, nrhs,
        D,
        E,
        B, ldb );
}

inline lapack_int LAPACKE_ptsv(
    lapack_int n, lapack_int nrhs,
    float* D,
    std::complex<float>* E,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cptsv(
        LAPACK_COL_MAJOR, n, nrhs,
        D,
        (lapack_complex_float*) E,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_ptsv(
    lapack_int n, lapack_int nrhs,
    double* D,
    std::complex<double>* E,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zptsv(
        LAPACK_COL_MAJOR, n, nrhs,
        D,
        (lapack_complex_double*) E,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_pttrf(
    lapack_int n,
    float* D,
    float* E )
{
    return LAPACKE_spttrf(
        n,
        D,
        E );
}

inline lapack_int LAPACKE_pttrf(
    lapack_int n,
    double* D,
    double* E )
{
    return LAPACKE_dpttrf(
        n,
        D,
        E );
}

inline lapack_int LAPACKE_pttrf(
    lapack_int n,
    float* D,
    std::complex<float>* E )
{
    return LAPACKE_cpttrf(
        n,
        D,
        (lapack_complex_float*) E );
}

inline lapack_int LAPACKE_pttrf(
    lapack_int n,
    double* D,
    std::complex<double>* E )
{
    return LAPACKE_zpttrf(
        n,
        D,
        (lapack_complex_double*) E );
}

// -----------------------------------------------------------------------------
// for [sd]pttrs, uplo is ignored
inline lapack_int LAPACKE_pttrs(
    char uplo, lapack_int n, lapack_int nrhs,
    float* D,
    float* E,
    float* B, lapack_int ldb )
{
    return LAPACKE_spttrs(
        LAPACK_COL_MAJOR, n, nrhs,
        D,
        E,
        B, ldb );
}

inline lapack_int LAPACKE_pttrs(
    char uplo, lapack_int n, lapack_int nrhs,
    double* D,
    double* E,
    double* B, lapack_int ldb )
{
    return LAPACKE_dpttrs(
        LAPACK_COL_MAJOR, n, nrhs,
        D,
        E,
        B, ldb );
}

inline lapack_int LAPACKE_pttrs(
    char uplo, lapack_int n, lapack_int nrhs,
    float* D,
    std::complex<float>* E,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cpttrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        D,
        (lapack_complex_float*) E,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_pttrs(
    char uplo, lapack_int n, lapack_int nrhs,
    double* D,
    std::complex<double>* E,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zpttrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        D,
        (lapack_complex_double*) E,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_spcon(
    char uplo, lapack_int n,
    float* AP,
    lapack_int* ipiv, float anorm,
    float* rcond )
{
    return LAPACKE_sspcon(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_spcon(
    char uplo, lapack_int n,
    double* AP,
    lapack_int* ipiv, double anorm,
    double* rcond )
{
    return LAPACKE_dspcon(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_spcon(
    char uplo, lapack_int n,
    std::complex<float>* AP,
    lapack_int* ipiv, float anorm,
    float* rcond )
{
    return LAPACKE_cspcon(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) AP,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_spcon(
    char uplo, lapack_int n,
    std::complex<double>* AP,
    lapack_int* ipiv, double anorm,
    double* rcond )
{
    return LAPACKE_zspcon(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) AP,
        ipiv, anorm,
        rcond );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_sprfs(
    char uplo, lapack_int n, lapack_int nrhs,
    float* AP,
    float* AFP,
    lapack_int* ipiv,
    float* B, lapack_int ldb,
    float* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_ssprfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        AFP,
        ipiv,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_sprfs(
    char uplo, lapack_int n, lapack_int nrhs,
    double* AP,
    double* AFP,
    lapack_int* ipiv,
    double* B, lapack_int ldb,
    double* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_dsprfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        AFP,
        ipiv,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_sprfs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* AP,
    std::complex<float>* AFP,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_csprfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) AFP,
        ipiv,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_sprfs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* AP,
    std::complex<double>* AFP,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_zsprfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) AFP,
        ipiv,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) X, ldx,
        ferr,
        berr );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_spsv(
    char uplo, lapack_int n, lapack_int nrhs,
    float* AP,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_sspsv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_spsv(
    char uplo, lapack_int n, lapack_int nrhs,
    double* AP,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dspsv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_spsv(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* AP,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cspsv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) AP,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_spsv(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* AP,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zspsv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) AP,
        ipiv,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_sptrf(
    char uplo, lapack_int n,
    float* AP,
    lapack_int* ipiv )
{
    return LAPACKE_ssptrf(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        ipiv );
}

inline lapack_int LAPACKE_sptrf(
    char uplo, lapack_int n,
    double* AP,
    lapack_int* ipiv )
{
    return LAPACKE_dsptrf(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        ipiv );
}

inline lapack_int LAPACKE_sptrf(
    char uplo, lapack_int n,
    std::complex<float>* AP,
    lapack_int* ipiv )
{
    return LAPACKE_csptrf(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) AP,
        ipiv );
}

inline lapack_int LAPACKE_sptrf(
    char uplo, lapack_int n,
    std::complex<double>* AP,
    lapack_int* ipiv )
{
    return LAPACKE_zsptrf(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) AP,
        ipiv );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_sptri(
    char uplo, lapack_int n,
    float* AP,
    lapack_int* ipiv )
{
    return LAPACKE_ssptri(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        ipiv );
}

inline lapack_int LAPACKE_sptri(
    char uplo, lapack_int n,
    double* AP,
    lapack_int* ipiv )
{
    return LAPACKE_dsptri(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        ipiv );
}

inline lapack_int LAPACKE_sptri(
    char uplo, lapack_int n,
    std::complex<float>* AP,
    lapack_int* ipiv )
{
    return LAPACKE_csptri(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) AP,
        ipiv );
}

inline lapack_int LAPACKE_sptri(
    char uplo, lapack_int n,
    std::complex<double>* AP,
    lapack_int* ipiv )
{
    return LAPACKE_zsptri(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) AP,
        ipiv );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_sptrs(
    char uplo, lapack_int n, lapack_int nrhs,
    float* AP,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_ssptrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_sptrs(
    char uplo, lapack_int n, lapack_int nrhs,
    double* AP,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dsptrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        AP,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_sptrs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* AP,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_csptrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) AP,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_sptrs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* AP,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zsptrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) AP,
        ipiv,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_sycon(
    char uplo, lapack_int n,
    float* A, lapack_int lda,
    lapack_int* ipiv, float anorm,
    float* rcond )
{
    return LAPACKE_ssycon(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_sycon(
    char uplo, lapack_int n,
    double* A, lapack_int lda,
    lapack_int* ipiv, double anorm,
    double* rcond )
{
    return LAPACKE_dsycon(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_sycon(
    char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv, float anorm,
    float* rcond )
{
    return LAPACKE_csycon(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) A, lda,
        ipiv, anorm,
        rcond );
}

inline lapack_int LAPACKE_sycon(
    char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv, double anorm,
    double* rcond )
{
    return LAPACKE_zsycon(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) A, lda,
        ipiv, anorm,
        rcond );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_syrfs(
    char uplo, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    float* AF, lapack_int ldaf,
    lapack_int* ipiv,
    float* B, lapack_int ldb,
    float* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_ssyrfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        AF, ldaf,
        ipiv,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_syrfs(
    char uplo, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    double* AF, lapack_int ldaf,
    lapack_int* ipiv,
    double* B, lapack_int ldb,
    double* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_dsyrfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        AF, ldaf,
        ipiv,
        B, ldb,
        X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_syrfs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* AF, lapack_int ldaf,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* X, lapack_int ldx,
    float* ferr,
    float* berr )
{
    return LAPACKE_csyrfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) AF, ldaf,
        ipiv,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) X, ldx,
        ferr,
        berr );
}

inline lapack_int LAPACKE_syrfs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* AF, lapack_int ldaf,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* X, lapack_int ldx,
    double* ferr,
    double* berr )
{
    return LAPACKE_zsyrfs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) AF, ldaf,
        ipiv,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) X, ldx,
        ferr,
        berr );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_sysv(
    char uplo, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_ssysv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_sysv(
    char uplo, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dsysv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_sysv(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_csysv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) A, lda,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_sysv(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zsysv(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) A, lda,
        ipiv,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
#if LAPACK_VERSION >= 30700
inline lapack_int LAPACKE_sysv_aa(
    char uplo, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_ssysv_aa(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_sysv_aa(
    char uplo, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dsysv_aa(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_sysv_aa(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_csysv_aa(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) A, lda,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_sysv_aa(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zsysv_aa(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) A, lda,
        ipiv,
        (lapack_complex_double*) B, ldb );
}
#endif // 30700

// -----------------------------------------------------------------------------
#if LAPACK_VERSION >= 30700
inline lapack_int LAPACKE_sysv_rk(
    char uplo, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    float* E,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_ssysv_rk(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        E,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_sysv_rk(
    char uplo, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    double* E,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dsysv_rk(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        E,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_sysv_rk(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* E,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_csysv_rk(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) E,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_sysv_rk(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* E,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zsysv_rk(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) E,
        ipiv,
        (lapack_complex_double*) B, ldb );
}
#endif // 30700

// -----------------------------------------------------------------------------
#if LAPACK_VERSION >= 30500
inline lapack_int LAPACKE_sysv_rook(
    char uplo, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_ssysv_rook(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_sysv_rook(
    char uplo, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dsysv_rook(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_sysv_rook(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_csysv_rook(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) A, lda,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_sysv_rook(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zsysv_rook(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) A, lda,
        ipiv,
        (lapack_complex_double*) B, ldb );
}
#endif // 30500

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_sytrf(
    char uplo, lapack_int n,
    float* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_ssytrf(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        ipiv );
}

inline lapack_int LAPACKE_sytrf(
    char uplo, lapack_int n,
    double* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_dsytrf(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        ipiv );
}

inline lapack_int LAPACKE_sytrf(
    char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_csytrf(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) A, lda,
        ipiv );
}

inline lapack_int LAPACKE_sytrf(
    char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_zsytrf(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) A, lda,
        ipiv );
}

// -----------------------------------------------------------------------------
// Fortran prototypes if not given via lapacke.h
extern "C" {

/* ----- symmetric indefinite factorization, Aasen's */
#ifndef LAPACK_ssytrf_aa
#define LAPACK_ssytrf_aa LAPACK_GLOBAL(ssytrf_aa,SSYTRF_AA)
void LAPACK_ssytrf_aa(
    char const* uplo,
    lapack_int const* n,
    float* a, lapack_int const* lda,
    lapack_int* ipiv,
    float* work, lapack_int const* lwork,
    lapack_int* info );
#endif

#ifndef LAPACK_dsytrf_aa
#define LAPACK_dsytrf_aa LAPACK_GLOBAL(dsytrf_aa,DSYTRF_AA)
void LAPACK_dsytrf_aa(
    char const* uplo,
    lapack_int const* n,
    double* a, lapack_int const* lda,
    lapack_int* ipiv,
    double* work, lapack_int const* lwork,
    lapack_int* info );
#endif

#ifndef LAPACK_csytrf_aa
#define LAPACK_csytrf_aa LAPACK_GLOBAL(csytrf_aa,CSYTRF_AA)
void LAPACK_csytrf_aa(
    char const* uplo,
    lapack_int const* n,
    lapack_complex_float* a, lapack_int const* lda,
    lapack_int* ipiv,
    lapack_complex_float* work, lapack_int const* lwork,
    lapack_int* info );
#endif

#ifndef LAPACK_zsytrf_aa
#define LAPACK_zsytrf_aa LAPACK_GLOBAL(zsytrf_aa,ZSYTRF_AA)
void LAPACK_zsytrf_aa(
    char const* uplo,
    lapack_int const* n,
    lapack_complex_double* a, lapack_int const* lda,
    lapack_int* ipiv,
    lapack_complex_double* work, lapack_int const* lwork,
    lapack_int* info );
#endif

#ifndef LAPACK_chetrf_aa
#define LAPACK_chetrf_aa LAPACK_GLOBAL(chetrf_aa,CHETRF_AA)
void LAPACK_chetrf_aa(
    char const* uplo,
    lapack_int const* n,
    lapack_complex_float* a, lapack_int const* lda,
    lapack_int* ipiv,
    lapack_complex_float* work, lapack_int const* lwork,
    lapack_int* info );
#endif

#ifndef LAPACK_zhetrf_aa
#define LAPACK_zhetrf_aa LAPACK_GLOBAL(zhetrf_aa,ZHETRF_AA)
void LAPACK_zhetrf_aa(
    char const* uplo,
    lapack_int const* n,
    lapack_complex_double* a, lapack_int const* lda,
    lapack_int* ipiv,
    lapack_complex_double* work, lapack_int const* lwork,
    lapack_int* info );
#endif

}  // extern "C"

// -----------------------------------------------------------------------------
// TODO: LAPACKE has a bug; use LAPACK for now
#if LAPACK_VERSION >= 30700
inline lapack_int LAPACKE_sytrf_aa(
    char uplo, lapack_int n,
    float* A, lapack_int lda,
    lapack_int* ipiv )
{
//  TODO Uncomment this when LAPACKE applies the bugfix
    // return LAPACKE_ssytrf_aa( LAPACK_COL_MAJOR, uplo, n, A, lda, ipiv );
    lapack_int info_ = 0;
    lapack_int lwork_ = n*128;
    std::vector< float > work( lwork_ );
    LAPACK_ssytrf_aa(
        &uplo, &n,
        A, &lda,
        ipiv,
        &work[0], &lwork_, &info_ );
    return info_;
}

inline lapack_int LAPACKE_sytrf_aa(
    char uplo, lapack_int n,
    double* A, lapack_int lda,
    lapack_int* ipiv )
{
//  TODO Uncomment this when LAPACKE applies the bugfix
    // return LAPACKE_dsytrf_aa( LAPACK_COL_MAJOR, uplo, n, A, lda, ipiv );
    lapack_int info_ = 0;
    lapack_int lwork_ = n*128;
    std::vector< double > work( lwork_ );
    LAPACK_dsytrf_aa(
        &uplo, &n,
        A, &lda,
        ipiv,
        &work[0], &lwork_, &info_ );
    return info_;
}

inline lapack_int LAPACKE_sytrf_aa(
    char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv )
{
//  TODO Uncomment this when LAPACKE applies the bugfix
    // return LAPACKE_csytrf_aa( LAPACK_COL_MAJOR, uplo, n, A, lda, ipiv );
    lapack_int info_ = 0;
    lapack_int lwork_ = n*128;
    std::vector< std::complex<float> > work( lwork_ );
    LAPACK_csytrf_aa(
        &uplo, &n,
        (lapack_complex_float*) A, &lda,
        ipiv,
        (lapack_complex_float*) &work[0], &lwork_, &info_ );
    return info_;
}

inline lapack_int LAPACKE_sytrf_aa(
    char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv )
{
//  TODO Uncomment this when LAPACKE applies the bugfix
    // return LAPACKE_zsytrf_aa( LAPACK_COL_MAJOR, uplo, n, A, lda, ipiv );
    lapack_int info_ = 0;
    lapack_int lwork_ = n*128;
    std::vector< std::complex<double> > work( lwork_ );
    LAPACK_zsytrf_aa(
        &uplo, &n,
        (lapack_complex_double*) A, &lda,
        ipiv,
        (lapack_complex_double*) &work[0], &lwork_, &info_ );
    return info_;
}
#endif // 30700

// -----------------------------------------------------------------------------
#if LAPACK_VERSION >= 30700
inline lapack_int LAPACKE_sytrf_rk(
    char uplo, lapack_int n,
    float* A, lapack_int lda,
    float* E,
    lapack_int* ipiv )
{
    return LAPACKE_ssytrf_rk(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        E,
        ipiv );
}

inline lapack_int LAPACKE_sytrf_rk(
    char uplo, lapack_int n,
    double* A, lapack_int lda,
    double* E,
    lapack_int* ipiv )
{
    return LAPACKE_dsytrf_rk(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        E,
        ipiv );
}

inline lapack_int LAPACKE_sytrf_rk(
    char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* E,
    lapack_int* ipiv )
{
    return LAPACKE_csytrf_rk(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) E,
        ipiv );
}

inline lapack_int LAPACKE_sytrf_rk(
    char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* E,
    lapack_int* ipiv )
{
    return LAPACKE_zsytrf_rk(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) E,
        ipiv );
}
#endif // 30700

// -----------------------------------------------------------------------------
#if LAPACK_VERSION >= 30500
inline lapack_int LAPACKE_sytrf_rook(
    char uplo, lapack_int n,
    float* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_ssytrf_rook(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        ipiv );
}

inline lapack_int LAPACKE_sytrf_rook(
    char uplo, lapack_int n,
    double* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_dsytrf_rook(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        ipiv );
}

inline lapack_int LAPACKE_sytrf_rook(
    char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_csytrf_rook(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) A, lda,
        ipiv );
}

inline lapack_int LAPACKE_sytrf_rook(
    char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_zsytrf_rook(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) A, lda,
        ipiv );
}
#endif // 30500

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_sytri(
    char uplo, lapack_int n,
    float* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_ssytri(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        ipiv );
}

inline lapack_int LAPACKE_sytri(
    char uplo, lapack_int n,
    double* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_dsytri(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        ipiv );
}

inline lapack_int LAPACKE_sytri(
    char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_csytri(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) A, lda,
        ipiv );
}

inline lapack_int LAPACKE_sytri(
    char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv )
{
    return LAPACKE_zsytri(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) A, lda,
        ipiv );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_sytrs(
    char uplo, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_ssytrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_sytrs(
    char uplo, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dsytrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_sytrs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_csytrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) A, lda,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_sytrs(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zsytrs(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) A, lda,
        ipiv,
        (lapack_complex_double*) B, ldb );
}

// -----------------------------------------------------------------------------
#if LAPACK_VERSION >= 30700
inline lapack_int LAPACKE_sytrs_aa(
    char uplo, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_ssytrs_aa(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_sytrs_aa(
    char uplo, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dsytrs_aa(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_sytrs_aa(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_csytrs_aa(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) A, lda,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_sytrs_aa(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zsytrs_aa(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) A, lda,
        ipiv,
        (lapack_complex_double*) B, ldb );
}
#endif // 30700

// -----------------------------------------------------------------------------
#if LAPACK_VERSION >= 30500
inline lapack_int LAPACKE_sytrs_rook(
    char uplo, lapack_int n, lapack_int nrhs,
    float* A, lapack_int lda,
    lapack_int* ipiv,
    float* B, lapack_int ldb )
{
    return LAPACKE_ssytrs_rook(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_sytrs_rook(
    char uplo, lapack_int n, lapack_int nrhs,
    double* A, lapack_int lda,
    lapack_int* ipiv,
    double* B, lapack_int ldb )
{
    return LAPACKE_dsytrs_rook(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        A, lda,
        ipiv,
        B, ldb );
}

inline lapack_int LAPACKE_sytrs_rook(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<float>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_csytrs_rook(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_float*) A, lda,
        ipiv,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_sytrs_rook(
    char uplo, lapack_int n, lapack_int nrhs,
    std::complex<double>* A, lapack_int lda,
    lapack_int* ipiv,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zsytrs_rook(
        LAPACK_COL_MAJOR, uplo, n, nrhs,
        (lapack_complex_double*) A, lda,
        ipiv,
        (lapack_complex_double*) B, ldb );
}
#endif // 30500

//------------------------------------------------------------------------------
inline lapack_int LAPACKE_tgexc(
    bool wantq, bool wantz, lapack_int n,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    float* Q, lapack_int ldq,
    float* Z, lapack_int ldz,
    lapack_int* ifst, lapack_int* ilst )
{
    return LAPACKE_stgexc(
        LAPACK_COL_MAJOR, wantq, wantz, n,
        A, lda,
        B, ldb,
        Q, ldq,
        Z, ldz,
        ifst, ilst );
}

inline lapack_int LAPACKE_tgexc(
    bool wantq, bool wantz, lapack_int n,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    double* Q, lapack_int ldq,
    double* Z, lapack_int ldz,
    lapack_int* ifst, lapack_int* ilst )
{
    return LAPACKE_dtgexc(
        LAPACK_COL_MAJOR, wantq, wantz, n,
        A, lda,
        B, ldb,
        Q, ldq,
        Z, ldz,
        ifst, ilst );
}

inline lapack_int LAPACKE_tgexc(
    bool wantq, bool wantz, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* Q, lapack_int ldq,
    std::complex<float>* Z, lapack_int ldz,
    lapack_int* ifst, lapack_int* ilst )
{
    // In complex, ifst, ilst are scalars instead of pointers.
    // Actually, ilst should be pointer; it is [in,out] in LAPACK.
    return LAPACKE_ctgexc(
        LAPACK_COL_MAJOR, wantq, wantz, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) Q, ldq,
        (lapack_complex_float*) Z, ldz,
        *ifst, *ilst );
}

inline lapack_int LAPACKE_tgexc(
    bool wantq, bool wantz, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* Q, lapack_int ldq,
    std::complex<double>* Z, lapack_int ldz,
    lapack_int* ifst, lapack_int* ilst )
{
    // In complex, ifst, ilst are scalars instead of pointers.
    // Actually, ilst should be pointer; it is [in,out] in LAPACK.
    return LAPACKE_ztgexc(
        LAPACK_COL_MAJOR, wantq, wantz, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) Q, ldq,
        (lapack_complex_double*) Z, ldz,
        *ifst, *ilst );
}

//------------------------------------------------------------------------------
inline lapack_int LAPACKE_tgsen(
    lapack_int ijob, bool wantq, bool wantz,
    lapack_int* select, lapack_int n,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    std::complex<float>* alpha,
    float* beta,
    float* Q, lapack_int ldq,
    float* Z, lapack_int ldz,
    lapack_int* sdim, float* pl, float* pr, float* dif )
{
    std::vector<float> alphar( n ), alphai( n );

    // query workspace
    float work_query;
    lapack_int liwork;
    lapack_int info = LAPACKE_stgsen_work(
        LAPACK_COL_MAJOR, ijob, wantq, wantz, select, n,
        A, lda, B, ldb,
        &alphar[0], &alphai[0], beta,
        Q, ldq, Z, ldz,
        sdim, pl, pr, dif,
        &work_query, -1, &liwork, -1 );
    assert( info == 0 );

    // allocate workspace
    // LAPACK <= 3.11 has query & documentation error in workspace size; add 1.
    lapack_int lwork = work_query + 1;
    assert( lwork >= 1 );
    assert( liwork >= 1 );
    std::vector<float> work( lwork );
    std::vector<lapack_int> iwork( liwork );

    info = LAPACKE_stgsen_work(
        LAPACK_COL_MAJOR, ijob, wantq, wantz, select, n,
        A, lda, B, ldb,
        &alphar[0], &alphai[0], beta,
        Q, ldq, Z, ldz,
        sdim, pl, pr, dif,
        &work[0], lwork, &iwork[0], liwork );

    // Merge split-complex representation.
    for (int64_t i = 0; i < n; ++i) {
        alpha[ i ] = std::complex<float>( alphar[ i ], alphai[ i ] );
    }
    return info;
}

inline lapack_int LAPACKE_tgsen(
    lapack_int ijob, bool wantq, bool wantz,
    lapack_int* select, lapack_int n,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    std::complex<double>* alpha,
    double* beta,
    double* Q, lapack_int ldq,
    double* Z, lapack_int ldz,
    lapack_int* sdim, double* pl, double* pr, double* dif )
{
    std::vector<double> alphar( n ), alphai( n );

    // query workspace
    double work_query;
    lapack_int liwork;
    lapack_int info = LAPACKE_dtgsen_work(
        LAPACK_COL_MAJOR, ijob, wantq, wantz, select, n,
        A, lda, B, ldb,
        &alphar[0], &alphai[0], beta,
        Q, ldq, Z, ldz,
        sdim, pl, pr, dif,
        &work_query, -1, &liwork, -1 );
    assert( info == 0 );

    // allocate workspace
    // LAPACK <= 3.11 has query & documentation error in workspace size; add 1.
    lapack_int lwork = work_query + 1;
    assert( lwork >= 1 );
    assert( liwork >= 1 );
    std::vector<double> work( lwork );
    std::vector<lapack_int> iwork( liwork );

    info = LAPACKE_dtgsen_work(
        LAPACK_COL_MAJOR, ijob, wantq, wantz, select, n,
        A, lda, B, ldb,
        &alphar[0], &alphai[0], beta,
        Q, ldq, Z, ldz,
        sdim, pl, pr, dif,
        &work[0], lwork, &iwork[0], liwork );

    // Merge split-complex representation.
    for (int64_t i = 0; i < n; ++i) {
        alpha[ i ] = std::complex<double>( alphar[ i ], alphai[ i ] );
    }
    return info;
}

inline lapack_int LAPACKE_tgsen(
    lapack_int ijob, bool wantq, bool wantz,
    lapack_int* select, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* alpha,
    std::complex<float>* beta,
    std::complex<float>* Q, lapack_int ldq,
    std::complex<float>* Z, lapack_int ldz,
    lapack_int* sdim, float* pl, float* pr, float* dif )
{
    // query workspace
    std::complex<float> work_query;
    lapack_int liwork;
    lapack_int info = LAPACKE_ctgsen_work(
        LAPACK_COL_MAJOR, ijob, wantq, wantz, select, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) alpha,
        (lapack_complex_float*) beta,
        (lapack_complex_float*) Q, ldq,
        (lapack_complex_float*) Z, ldz,
        sdim, pl, pr, dif,
        (lapack_complex_float*) &work_query, -1, &liwork, -1 );
    assert( info == 0 );

    // allocate workspace
    // LAPACK <= 3.11 has query & documentation error in workspace size; add 1.
    lapack_int lwork = real( work_query ) + 1;
    assert( lwork >= 1 );
    assert( liwork >= 1 );
    std::vector<lapack_complex_float> work( lwork );
    std::vector<lapack_int> iwork( liwork );

    return LAPACKE_ctgsen_work(
        LAPACK_COL_MAJOR, ijob, wantq, wantz, select, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) alpha,
        (lapack_complex_float*) beta,
        (lapack_complex_float*) Q, ldq,
        (lapack_complex_float*) Z, ldz,
        sdim, pl, pr, dif,
        &work[0], lwork, &iwork[0], liwork );
}

inline lapack_int LAPACKE_tgsen(
    lapack_int ijob, bool wantq, bool wantz,
    lapack_int* select, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* alpha,
    std::complex<double>* beta,
    std::complex<double>* Q, lapack_int ldq,
    std::complex<double>* Z, lapack_int ldz,
    lapack_int* sdim, double* pl, double* pr, double* dif )
{
    // query workspace
    std::complex<double> work_query;
    lapack_int liwork;
    lapack_int info = LAPACKE_ztgsen_work(
        LAPACK_COL_MAJOR, ijob, wantq, wantz, select, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) alpha,
        (lapack_complex_double*) beta,
        (lapack_complex_double*) Q, ldq,
        (lapack_complex_double*) Z, ldz,
        sdim, pl, pr, dif,
        (lapack_complex_double*) &work_query, -1, &liwork, -1 );
    assert( info == 0 );

    // allocate workspace
    // LAPACK <= 3.11 has query & documentation error in workspace size; add 1.
    lapack_int lwork = real( work_query ) + 1;
    assert( lwork >= 1 );
    assert( liwork >= 1 );
    std::vector<lapack_complex_double> work( lwork );
    std::vector<lapack_int> iwork( liwork );

    return LAPACKE_ztgsen_work(
        LAPACK_COL_MAJOR, ijob, wantq, wantz, select, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) alpha,
        (lapack_complex_double*) beta,
        (lapack_complex_double*) Q, ldq,
        (lapack_complex_double*) Z, ldz,
        sdim, pl, pr, dif,
        &work[0], lwork, &iwork[0], liwork );
}

// -----------------------------------------------------------------------------
#if LAPACK_VERSION >= 30700

// Fortran prototypes if not given via lapacke.h
extern "C" {

/* ----- LQ factorization of triangular A and pentagonal B */
#ifndef LAPACK_stplqt
#define LAPACK_stplqt LAPACK_GLOBAL(stplqt,STPLQT)
void LAPACK_stplqt(
    lapack_int const* m, lapack_int const* n, lapack_int const* l,
    lapack_int const* mb,
    float* A, lapack_int const* lda,
    float* B, lapack_int const* ldb,
    float* T, lapack_int const* ldt,
    float* work,
    lapack_int* info );
#endif

#ifndef LAPACK_dtplqt
#define LAPACK_dtplqt LAPACK_GLOBAL(dtplqt,DTPLQT)
void LAPACK_dtplqt(
    lapack_int const* m, lapack_int const* n, lapack_int const* l,
    lapack_int const* mb,
    double* A, lapack_int const* lda,
    double* B, lapack_int const* ldb,
    double* T, lapack_int const* ldt,
    double* work,
    lapack_int* info );
#endif

#ifndef LAPACK_ctplqt
#define LAPACK_ctplqt LAPACK_GLOBAL(ctplqt,CTPLQT)
void LAPACK_ctplqt(
    lapack_int const* m, lapack_int const* n, lapack_int const* l,
    lapack_int const* mb,
    lapack_complex_float* A, lapack_int const* lda,
    lapack_complex_float* B, lapack_int const* ldb,
    lapack_complex_float* T, lapack_int const* ldt,
    lapack_complex_float* work,
    lapack_int* info );
#endif

#ifndef LAPACK_ztplqt
#define LAPACK_ztplqt LAPACK_GLOBAL(ztplqt,ZTPLQT)
void LAPACK_ztplqt(
    lapack_int const* m, lapack_int const* n, lapack_int const* l,
    lapack_int const* mb,
    lapack_complex_double* A, lapack_int const* lda,
    lapack_complex_double* B, lapack_int const* ldb,
    lapack_complex_double* T, lapack_int const* ldt,
    lapack_complex_double* work,
    lapack_int* info );
#endif

}  // extern "C"

// --------------------
// wrappers around LAPACK (not in LAPACKE)
inline lapack_int LAPACKE_tplqt(
    lapack_int m, lapack_int n, lapack_int l, lapack_int mb,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    float* T, lapack_int ldt )
{
    std::vector<float> work( mb*m );
    lapack_int info = 0;
    LAPACK_stplqt(
        &m, &n, &l, &mb,
        A, &lda,
        B, &ldb,
        T, &ldt,
        work.data(),
        &info );
    return info;
}

inline lapack_int LAPACKE_tplqt(
    lapack_int m, lapack_int n, lapack_int l, lapack_int mb,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    double* T, lapack_int ldt )
{
    std::vector<double> work( mb*m );
    lapack_int info = 0;
    LAPACK_dtplqt(
        &m, &n, &l, &mb,
        A, &lda,
        B, &ldb,
        T, &ldt,
        work.data(),
        &info );
    return info;
}

inline lapack_int LAPACKE_tplqt(
    lapack_int m, lapack_int n, lapack_int l, lapack_int mb,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* T, lapack_int ldt )
{
    std::vector<lapack_complex_float> work( mb*m );
    lapack_int info = 0;
    LAPACK_ctplqt(
        &m, &n, &l, &mb,
        (lapack_complex_float*) A, &lda,
        (lapack_complex_float*) B, &ldb,
        (lapack_complex_float*) T, &ldt,
        work.data(),
        &info );
    return info;
}

inline lapack_int LAPACKE_tplqt(
    lapack_int m, lapack_int n, lapack_int l, lapack_int mb,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* T, lapack_int ldt )
{
    std::vector<lapack_complex_double> work( mb*m );
    lapack_int info = 0;
    LAPACK_ztplqt(
        &m, &n, &l, &mb,
        (lapack_complex_double*) A, &lda,
        (lapack_complex_double*) B, &ldb,
        (lapack_complex_double*) T, &ldt,
        work.data(),
        &info );
    return info;
}
#endif // 30700

// -----------------------------------------------------------------------------
// Fortran prototypes if not given via lapacke.h
extern "C" {

/* ----- LQ factorization of triangular A and pentagonal B */
#ifndef LAPACK_stplqt2
#define LAPACK_stplqt2 LAPACK_GLOBAL(stplqt2,STPLQT2)
void LAPACK_stplqt2(
    lapack_int const* m, lapack_int const* n, lapack_int const* l,
    float* A, lapack_int const* lda,
    float* B, lapack_int const* ldb,
    float* T, lapack_int const* ldt,
    lapack_int* info );
#endif

#ifndef LAPACK_dtplqt2
#define LAPACK_dtplqt2 LAPACK_GLOBAL(dtplqt2,DTPLQT2)
void LAPACK_dtplqt2(
    lapack_int const* m, lapack_int const* n, lapack_int const* l,
    double* A, lapack_int const* lda,
    double* B, lapack_int const* ldb,
    double* T, lapack_int const* ldt,
    lapack_int* info );
#endif

#ifndef LAPACK_ctplqt2
#define LAPACK_ctplqt2 LAPACK_GLOBAL(ctplqt2,CTPLQT2)
void LAPACK_ctplqt2(
    lapack_int const* m, lapack_int const* n, lapack_int const* l,
    lapack_complex_float* A, lapack_int const* lda,
    lapack_complex_float* B, lapack_int const* ldb,
    lapack_complex_float* T, lapack_int const* ldt,
    lapack_int* info );
#endif

#ifndef LAPACK_ztplqt2
#define LAPACK_ztplqt2 LAPACK_GLOBAL(ztplqt2,ZTPLQT2)
void LAPACK_ztplqt2(
    lapack_int const* m, lapack_int const* n, lapack_int const* l,
    lapack_complex_double* A, lapack_int const* lda,
    lapack_complex_double* B, lapack_int const* ldb,
    lapack_complex_double* T, lapack_int const* ldt,
    lapack_int* info );
#endif

}  // extern "C"

// --------------------
// wrappers around LAPACK (not in LAPACKE)
#if LAPACK_VERSION >= 30700
inline lapack_int LAPACKE_tplqt2(
    lapack_int m, lapack_int n, lapack_int l,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    float* T, lapack_int ldt )
{
    lapack_int info = 0;
    LAPACK_stplqt2(
        &m, &n, &l,
        A, &lda,
        B, &ldb,
        T, &ldt,
        &info );
    return info;
}

inline lapack_int LAPACKE_tplqt2(
    lapack_int m, lapack_int n, lapack_int l,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    double* T, lapack_int ldt )
{
    lapack_int info = 0;
    LAPACK_dtplqt2(
        &m, &n, &l,
        A, &lda,
        B, &ldb,
        T, &ldt,
        &info );
    return info;
}

inline lapack_int LAPACKE_tplqt2(
    lapack_int m, lapack_int n, lapack_int l,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* T, lapack_int ldt )
{
    lapack_int info = 0;
    LAPACK_ctplqt2(
        &m, &n, &l,
        (lapack_complex_float*) A, &lda,
        (lapack_complex_float*) B, &ldb,
        (lapack_complex_float*) T, &ldt,
        &info );
    return info;
}

inline lapack_int LAPACKE_tplqt2(
    lapack_int m, lapack_int n, lapack_int l,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* T, lapack_int ldt )
{
    lapack_int info = 0;
    LAPACK_ztplqt2(
        &m, &n, &l,
        (lapack_complex_double*) A, &lda,
        (lapack_complex_double*) B, &ldb,
        (lapack_complex_double*) T, &ldt,
        &info );
    return info;
}
#endif // 30700

// -----------------------------------------------------------------------------
// Fortran prototypes if not given via lapacke.h
extern "C" {

/* ----- multiply by Q from tplqt */
#ifndef LAPACK_stpmlqt
#define LAPACK_stpmlqt LAPACK_GLOBAL(stpmlqt,STPMLQT)
void LAPACK_stpmlqt(
    char const* side, char const* trans,
    lapack_int const* m, lapack_int const* n, lapack_int const* k,
    lapack_int const* l, lapack_int const* mb,
    float const* V, lapack_int const* ldv,
    float const* T, lapack_int const* ldt,
    float* A, lapack_int const* lda,
    float* B, lapack_int const* ldb,
    float* work,
    lapack_int* info );
#endif

#ifndef LAPACK_dtpmlqt
#define LAPACK_dtpmlqt LAPACK_GLOBAL(dtpmlqt,DTPMLQT)
void LAPACK_dtpmlqt(
    char const* side, char const* trans,
    lapack_int const* m, lapack_int const* n, lapack_int const* k,
    lapack_int const* l, lapack_int const* mb,
    double const* V, lapack_int const* ldv,
    double const* T, lapack_int const* ldt,
    double* A, lapack_int const* lda,
    double* B, lapack_int const* ldb,
    double* work,
    lapack_int* info );
#endif

#ifndef LAPACK_ctpmlqt
#define LAPACK_ctpmlqt LAPACK_GLOBAL(ctpmlqt,CTPMLQT)
void LAPACK_ctpmlqt(
    char const* side, char const* trans,
    lapack_int const* m, lapack_int const* n, lapack_int const* k,
    lapack_int const* l, lapack_int const* mb,
    lapack_complex_float const* V, lapack_int const* ldv,
    lapack_complex_float const* T, lapack_int const* ldt,
    lapack_complex_float* A, lapack_int const* lda,
    lapack_complex_float* B, lapack_int const* ldb,
    lapack_complex_float* work,
    lapack_int* info );
#endif

#ifndef LAPACK_ztpmlqt
#define LAPACK_ztpmlqt LAPACK_GLOBAL(ztpmlqt,ZTPMLQT)
void LAPACK_ztpmlqt(
    char const* side, char const* trans,
    lapack_int const* m, lapack_int const* n, lapack_int const* k,
    lapack_int const* l, lapack_int const* mb,
    lapack_complex_double const* V, lapack_int const* ldv,
    lapack_complex_double const* T, lapack_int const* ldt,
    lapack_complex_double* A, lapack_int const* lda,
    lapack_complex_double* B, lapack_int const* ldb,
    lapack_complex_double* work,
    lapack_int* info );
#endif

}  // extern "C"

// --------------------
// wrappers around LAPACK (not in LAPACKE)
#if LAPACK_VERSION >= 30700
inline lapack_int LAPACKE_tpmlqt(
    char side, char trans,
    lapack_int m, lapack_int n, lapack_int k, lapack_int l, lapack_int mb,
    float* V, lapack_int ldv,
    float* T, lapack_int ldt,
    float* A, lapack_int lda,
    float* B, lapack_int ldb )
{
    if (trans == 'C')
        trans = 'T';
    std::vector<float> work( side == 'L' ? n*mb : m*mb );
    lapack_int info = 0;
    LAPACK_stpmlqt(
        &side, &trans, &m, &n, &k, &l, &mb,
        V, &ldv,
        T, &ldt,
        A, &lda,
        B, &ldb,
        work.data(),
        &info );
    return info;
}

inline lapack_int LAPACKE_tpmlqt(
    char side, char trans,
    lapack_int m, lapack_int n, lapack_int k, lapack_int l, lapack_int mb,
    double* V, lapack_int ldv,
    double* T, lapack_int ldt,
    double* A, lapack_int lda,
    double* B, lapack_int ldb )
{
    if (trans == 'C')
        trans = 'T';
    std::vector<double> work( side == 'L' ? n*mb : m*mb );
    lapack_int info = 0;
    LAPACK_dtpmlqt(
        &side, &trans, &m, &n, &k, &l, &mb,
        V, &ldv,
        T, &ldt,
        A, &lda,
        B, &ldb,
        work.data(),
        &info );
    return info;
}

inline lapack_int LAPACKE_tpmlqt(
    char side, char trans,
    lapack_int m, lapack_int n, lapack_int k, lapack_int l, lapack_int mb,
    std::complex<float>* V, lapack_int ldv,
    std::complex<float>* T, lapack_int ldt,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb )
{
    std::vector<lapack_complex_float> work( side == 'L' ? n*mb : m*mb );
    lapack_int info = 0;
    LAPACK_ctpmlqt(
        &side, &trans, &m, &n, &k, &l, &mb,
        (lapack_complex_float*) V, &ldv,
        (lapack_complex_float*) T, &ldt,
        (lapack_complex_float*) A, &lda,
        (lapack_complex_float*) B, &ldb,
        work.data(),
        &info );
    return info;
}

inline lapack_int LAPACKE_tpmlqt(
    char side, char trans,
    lapack_int m, lapack_int n, lapack_int k, lapack_int l, lapack_int mb,
    std::complex<double>* V, lapack_int ldv,
    std::complex<double>* T, lapack_int ldt,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb )
{
    std::vector<lapack_complex_double> work( side == 'L' ? n*mb : m*mb );
    lapack_int info = 0;
    LAPACK_ztpmlqt(
        &side, &trans, &m, &n, &k, &l, &mb,
        (lapack_complex_double*) V, &ldv,
        (lapack_complex_double*) T, &ldt,
        (lapack_complex_double*) A, &lda,
        (lapack_complex_double*) B, &ldb,
        work.data(),
        &info );
    return info;
}
#endif // 30700

// -----------------------------------------------------------------------------
#if LAPACK_VERSION >= 30400
inline lapack_int LAPACKE_tpmqrt(
    char side, char trans, lapack_int m, lapack_int n, lapack_int k, lapack_int l, lapack_int nb,
    float* V, lapack_int ldv,
    float* T, lapack_int ldt,
    float* A, lapack_int lda,
    float* B, lapack_int ldb )
{
    if (trans == 'C')
        trans = 'T';
    return LAPACKE_stpmqrt(
        LAPACK_COL_MAJOR, side, trans, m, n, k, l, nb,
        V, ldv,
        T, ldt,
        A, lda,
        B, ldb );
}

inline lapack_int LAPACKE_tpmqrt(
    char side, char trans, lapack_int m, lapack_int n, lapack_int k, lapack_int l, lapack_int nb,
    double* V, lapack_int ldv,
    double* T, lapack_int ldt,
    double* A, lapack_int lda,
    double* B, lapack_int ldb )
{
    if (trans == 'C')
        trans = 'T';
    return LAPACKE_dtpmqrt(
        LAPACK_COL_MAJOR, side, trans, m, n, k, l, nb,
        V, ldv,
        T, ldt,
        A, lda,
        B, ldb );
}

inline lapack_int LAPACKE_tpmqrt(
    char side, char trans, lapack_int m, lapack_int n, lapack_int k, lapack_int l, lapack_int nb,
    std::complex<float>* V, lapack_int ldv,
    std::complex<float>* T, lapack_int ldt,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_ctpmqrt(
        LAPACK_COL_MAJOR, side, trans, m, n, k, l, nb,
        (lapack_complex_float*) V, ldv,
        (lapack_complex_float*) T, ldt,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_tpmqrt(
    char side, char trans, lapack_int m, lapack_int n, lapack_int k, lapack_int l, lapack_int nb,
    std::complex<double>* V, lapack_int ldv,
    std::complex<double>* T, lapack_int ldt,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_ztpmqrt(
        LAPACK_COL_MAJOR, side, trans, m, n, k, l, nb,
        (lapack_complex_double*) V, ldv,
        (lapack_complex_double*) T, ldt,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb );
}
#endif // 30400

// -----------------------------------------------------------------------------
#if LAPACK_VERSION >= 30400
inline lapack_int LAPACKE_tpqrt(
    lapack_int m, lapack_int n, lapack_int l, lapack_int nb,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    float* T, lapack_int ldt )
{
    return LAPACKE_stpqrt(
        LAPACK_COL_MAJOR, m, n, l, nb,
        A, lda,
        B, ldb,
        T, ldt );
}

inline lapack_int LAPACKE_tpqrt(
    lapack_int m, lapack_int n, lapack_int l, lapack_int nb,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    double* T, lapack_int ldt )
{
    return LAPACKE_dtpqrt(
        LAPACK_COL_MAJOR, m, n, l, nb,
        A, lda,
        B, ldb,
        T, ldt );
}

inline lapack_int LAPACKE_tpqrt(
    lapack_int m, lapack_int n, lapack_int l, lapack_int nb,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* T, lapack_int ldt )
{
    return LAPACKE_ctpqrt(
        LAPACK_COL_MAJOR, m, n, l, nb,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) T, ldt );
}

inline lapack_int LAPACKE_tpqrt(
    lapack_int m, lapack_int n, lapack_int l, lapack_int nb,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* T, lapack_int ldt )
{
    return LAPACKE_ztpqrt(
        LAPACK_COL_MAJOR, m, n, l, nb,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) T, ldt );
}
#endif // 30400

// -----------------------------------------------------------------------------
#if LAPACK_VERSION >= 30400
inline lapack_int LAPACKE_tpqrt2(
    lapack_int m, lapack_int n, lapack_int l,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    float* T, lapack_int ldt )
{
    return LAPACKE_stpqrt2(
        LAPACK_COL_MAJOR, m, n, l,
        A, lda,
        B, ldb,
        T, ldt );
}

inline lapack_int LAPACKE_tpqrt2(
    lapack_int m, lapack_int n, lapack_int l,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    double* T, lapack_int ldt )
{
    return LAPACKE_dtpqrt2(
        LAPACK_COL_MAJOR, m, n, l,
        A, lda,
        B, ldb,
        T, ldt );
}

inline lapack_int LAPACKE_tpqrt2(
    lapack_int m, lapack_int n, lapack_int l,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* T, lapack_int ldt )
{
    return LAPACKE_ctpqrt2(
        LAPACK_COL_MAJOR, m, n, l,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb,
        (lapack_complex_float*) T, ldt );
}

inline lapack_int LAPACKE_tpqrt2(
    lapack_int m, lapack_int n, lapack_int l,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* T, lapack_int ldt )
{
    return LAPACKE_ztpqrt2(
        LAPACK_COL_MAJOR, m, n, l,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb,
        (lapack_complex_double*) T, ldt );
}
#endif // 30400

// -----------------------------------------------------------------------------
#if LAPACK_VERSION >= 30400
inline lapack_int LAPACKE_tprfb(
    char side, char trans, char direction, char storev, lapack_int m, lapack_int n, lapack_int k, lapack_int l,
    float* V, lapack_int ldv,
    float* T, lapack_int ldt,
    float* A, lapack_int lda,
    float* B, lapack_int ldb )
{
    return LAPACKE_stprfb(
        LAPACK_COL_MAJOR, side, trans, direction, storev, m, n, k, l,
        V, ldv,
        T, ldt,
        A, lda,
        B, ldb );
}

inline lapack_int LAPACKE_tprfb(
    char side, char trans, char direction, char storev, lapack_int m, lapack_int n, lapack_int k, lapack_int l,
    double* V, lapack_int ldv,
    double* T, lapack_int ldt,
    double* A, lapack_int lda,
    double* B, lapack_int ldb )
{
    return LAPACKE_dtprfb(
        LAPACK_COL_MAJOR, side, trans, direction, storev, m, n, k, l,
        V, ldv,
        T, ldt,
        A, lda,
        B, ldb );
}

inline lapack_int LAPACKE_tprfb(
    char side, char trans, char direction, char storev, lapack_int m, lapack_int n, lapack_int k, lapack_int l,
    std::complex<float>* V, lapack_int ldv,
    std::complex<float>* T, lapack_int ldt,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_ctprfb(
        LAPACK_COL_MAJOR, side, trans, direction, storev, m, n, k, l,
        (lapack_complex_float*) V, ldv,
        (lapack_complex_float*) T, ldt,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) B, ldb );
}

inline lapack_int LAPACKE_tprfb(
    char side, char trans, char direction, char storev, lapack_int m, lapack_int n, lapack_int k, lapack_int l,
    std::complex<double>* V, lapack_int ldv,
    std::complex<double>* T, lapack_int ldt,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_ztprfb(
        LAPACK_COL_MAJOR, side, trans, direction, storev, m, n, k, l,
        (lapack_complex_double*) V, ldv,
        (lapack_complex_double*) T, ldt,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) B, ldb );
}
#endif // 30400

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_unghr(
    lapack_int n, lapack_int ilo, lapack_int ihi,
    float* A, lapack_int lda,
    float* tau )
{
    return LAPACKE_sorghr(
        LAPACK_COL_MAJOR, n, ilo, ihi,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_unghr(
    lapack_int n, lapack_int ilo, lapack_int ihi,
    double* A, lapack_int lda,
    double* tau )
{
    return LAPACKE_dorghr(
        LAPACK_COL_MAJOR, n, ilo, ihi,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_unghr(
    lapack_int n, lapack_int ilo, lapack_int ihi,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* tau )
{
    return LAPACKE_cunghr(
        LAPACK_COL_MAJOR, n, ilo, ihi,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) tau );
}

inline lapack_int LAPACKE_unghr(
    lapack_int n, lapack_int ilo, lapack_int ihi,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* tau )
{
    return LAPACKE_zunghr(
        LAPACK_COL_MAJOR, n, ilo, ihi,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) tau );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_unglq(
    lapack_int m, lapack_int n, lapack_int k,
    float* A, lapack_int lda,
    float* tau )
{
    return LAPACKE_sorglq(
        LAPACK_COL_MAJOR, m, n, k,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_unglq(
    lapack_int m, lapack_int n, lapack_int k,
    double* A, lapack_int lda,
    double* tau )
{
    return LAPACKE_dorglq(
        LAPACK_COL_MAJOR, m, n, k,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_unglq(
    lapack_int m, lapack_int n, lapack_int k,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* tau )
{
    return LAPACKE_cunglq(
        LAPACK_COL_MAJOR, m, n, k,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) tau );
}

inline lapack_int LAPACKE_unglq(
    lapack_int m, lapack_int n, lapack_int k,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* tau )
{
    return LAPACKE_zunglq(
        LAPACK_COL_MAJOR, m, n, k,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) tau );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_ungql(
    lapack_int m, lapack_int n, lapack_int k,
    float* A, lapack_int lda,
    float* tau )
{
    return LAPACKE_sorgql(
        LAPACK_COL_MAJOR, m, n, k,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_ungql(
    lapack_int m, lapack_int n, lapack_int k,
    double* A, lapack_int lda,
    double* tau )
{
    return LAPACKE_dorgql(
        LAPACK_COL_MAJOR, m, n, k,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_ungql(
    lapack_int m, lapack_int n, lapack_int k,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* tau )
{
    return LAPACKE_cungql(
        LAPACK_COL_MAJOR, m, n, k,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) tau );
}

inline lapack_int LAPACKE_ungql(
    lapack_int m, lapack_int n, lapack_int k,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* tau )
{
    return LAPACKE_zungql(
        LAPACK_COL_MAJOR, m, n, k,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) tau );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_ungqr(
    lapack_int m, lapack_int n, lapack_int k,
    float* A, lapack_int lda,
    float* tau )
{
    return LAPACKE_sorgqr(
        LAPACK_COL_MAJOR, m, n, k,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_ungqr(
    lapack_int m, lapack_int n, lapack_int k,
    double* A, lapack_int lda,
    double* tau )
{
    return LAPACKE_dorgqr(
        LAPACK_COL_MAJOR, m, n, k,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_ungqr(
    lapack_int m, lapack_int n, lapack_int k,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* tau )
{
    return LAPACKE_cungqr(
        LAPACK_COL_MAJOR, m, n, k,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) tau );
}

inline lapack_int LAPACKE_ungqr(
    lapack_int m, lapack_int n, lapack_int k,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* tau )
{
    return LAPACKE_zungqr(
        LAPACK_COL_MAJOR, m, n, k,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) tau );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_ungrq(
    lapack_int m, lapack_int n, lapack_int k,
    float* A, lapack_int lda,
    float* tau )
{
    return LAPACKE_sorgrq(
        LAPACK_COL_MAJOR, m, n, k,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_ungrq(
    lapack_int m, lapack_int n, lapack_int k,
    double* A, lapack_int lda,
    double* tau )
{
    return LAPACKE_dorgrq(
        LAPACK_COL_MAJOR, m, n, k,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_ungrq(
    lapack_int m, lapack_int n, lapack_int k,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* tau )
{
    return LAPACKE_cungrq(
        LAPACK_COL_MAJOR, m, n, k,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) tau );
}

inline lapack_int LAPACKE_ungrq(
    lapack_int m, lapack_int n, lapack_int k,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* tau )
{
    return LAPACKE_zungrq(
        LAPACK_COL_MAJOR, m, n, k,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) tau );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_ungtr(
    char uplo, lapack_int n,
    float* A, lapack_int lda,
    float* tau )
{
    return LAPACKE_sorgtr(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_ungtr(
    char uplo, lapack_int n,
    double* A, lapack_int lda,
    double* tau )
{
    return LAPACKE_dorgtr(
        LAPACK_COL_MAJOR, uplo, n,
        A, lda,
        tau );
}

inline lapack_int LAPACKE_ungtr(
    char uplo, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* tau )
{
    return LAPACKE_cungtr(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) tau );
}

inline lapack_int LAPACKE_ungtr(
    char uplo, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* tau )
{
    return LAPACKE_zungtr(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) tau );
}

// -----------------------------------------------------------------------------
// LAPACKE_*unhr_col only in Intel MKL, not yet Netlib LAPACK as of 3.10.
#if LAPACK_VERSION >= 30900 && defined( LAPACK_HAVE_MKL ) // >= 3.9.0

inline lapack_int LAPACKE_orhr_col(
    lapack_int m, lapack_int n, lapack_int nb,
    float* A, lapack_int lda,
    float* T, lapack_int ldt,
    float* D )
{
    return LAPACKE_sorhr_col(
        LAPACK_COL_MAJOR, m, n, nb,
        A, lda,
        T, ldt,
        D );
}

inline lapack_int LAPACKE_orhr_col(
    lapack_int m, lapack_int n, lapack_int nb,
    double* A, lapack_int lda,
    double* T, lapack_int ldt,
    double* D )
{
    return LAPACKE_dorhr_col(
        LAPACK_COL_MAJOR, m, n, nb,
        A, lda,
        T, ldt,
        D );
}

// In real, unhr is an alias for orhr.
inline lapack_int LAPACKE_unhr_col(
    lapack_int m, lapack_int n, lapack_int nb,
    float* A, lapack_int lda,
    float* T, lapack_int ldt,
    float* D )
{
    return LAPACKE_orhr_col( m, n, nb, A, lda, T, ldt, D );
}

// In real, unhr is an alias for orhr.
inline lapack_int LAPACKE_unhr_col(
    lapack_int m, lapack_int n, lapack_int nb,
    double* A, lapack_int lda,
    double* T, lapack_int ldt,
    double* D )
{
    return LAPACKE_orhr_col( m, n, nb, A, lda, T, ldt, D );
}

inline lapack_int LAPACKE_unhr_col(
    lapack_int m, lapack_int n, lapack_int nb,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* T, lapack_int ldt,
    std::complex<float>* D )
{
    return LAPACKE_cunhr_col(
        LAPACK_COL_MAJOR, m, n, nb,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) T, ldt,
        (lapack_complex_float*) D );
}

inline lapack_int LAPACKE_unhr_col(
    lapack_int m, lapack_int n, lapack_int nb,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* T, lapack_int ldt,
    std::complex<double>* D )
{
    return LAPACKE_zunhr_col(
        LAPACK_COL_MAJOR, m, n, nb,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) T, ldt,
        (lapack_complex_double*) D );
}

#endif  // 3.9.0 and LAPACK_HAVE_MKL


// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_unmhr(
    char side, char trans, lapack_int m, lapack_int n, lapack_int ilo, lapack_int ihi,
    float* A, lapack_int lda,
    float* tau,
    float* C, lapack_int ldc )
{
    if (trans == 'C')
        trans = 'T';
    return LAPACKE_sormhr(
        LAPACK_COL_MAJOR, side, trans, m, n, ilo, ihi,
        A, lda,
        tau,
        C, ldc );
}

inline lapack_int LAPACKE_unmhr(
    char side, char trans, lapack_int m, lapack_int n, lapack_int ilo, lapack_int ihi,
    double* A, lapack_int lda,
    double* tau,
    double* C, lapack_int ldc )
{
    if (trans == 'C')
        trans = 'T';
    return LAPACKE_dormhr(
        LAPACK_COL_MAJOR, side, trans, m, n, ilo, ihi,
        A, lda,
        tau,
        C, ldc );
}

inline lapack_int LAPACKE_unmhr(
    char side, char trans, lapack_int m, lapack_int n, lapack_int ilo, lapack_int ihi,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* tau,
    std::complex<float>* C, lapack_int ldc )
{
    return LAPACKE_cunmhr(
        LAPACK_COL_MAJOR, side, trans, m, n, ilo, ihi,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) C, ldc );
}

inline lapack_int LAPACKE_unmhr(
    char side, char trans, lapack_int m, lapack_int n, lapack_int ilo, lapack_int ihi,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* tau,
    std::complex<double>* C, lapack_int ldc )
{
    return LAPACKE_zunmhr(
        LAPACK_COL_MAJOR, side, trans, m, n, ilo, ihi,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) C, ldc );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_unmtr(
    char side, char uplo, char trans, lapack_int m, lapack_int n,
    float* A, lapack_int lda,
    float* tau,
    float* C, lapack_int ldc )
{
    if (trans == 'C')
        trans = 'T';
    return LAPACKE_sormtr(
        LAPACK_COL_MAJOR, side, uplo, trans, m, n,
        A, lda,
        tau,
        C, ldc );
}

inline lapack_int LAPACKE_unmtr(
    char side, char uplo, char trans, lapack_int m, lapack_int n,
    double* A, lapack_int lda,
    double* tau,
    double* C, lapack_int ldc )
{
    if (trans == 'C')
        trans = 'T';
    return LAPACKE_dormtr(
        LAPACK_COL_MAJOR, side, uplo, trans, m, n,
        A, lda,
        tau,
        C, ldc );
}

inline lapack_int LAPACKE_unmtr(
    char side, char uplo, char trans, lapack_int m, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* tau,
    std::complex<float>* C, lapack_int ldc )
{
    return LAPACKE_cunmtr(
        LAPACK_COL_MAJOR, side, uplo, trans, m, n,
        (lapack_complex_float*) A, lda,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) C, ldc );
}

inline lapack_int LAPACKE_unmtr(
    char side, char uplo, char trans, lapack_int m, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* tau,
    std::complex<double>* C, lapack_int ldc )
{
    return LAPACKE_zunmtr(
        LAPACK_COL_MAJOR, side, uplo, trans, m, n,
        (lapack_complex_double*) A, lda,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) C, ldc );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_upgtr(
    char uplo, lapack_int n,
    float* AP,
    float* tau,
    float* Q, lapack_int ldq )
{
    return LAPACKE_sopgtr(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        tau,
        Q, ldq );
}

inline lapack_int LAPACKE_upgtr(
    char uplo, lapack_int n,
    double* AP,
    double* tau,
    double* Q, lapack_int ldq )
{
    return LAPACKE_dopgtr(
        LAPACK_COL_MAJOR, uplo, n,
        AP,
        tau,
        Q, ldq );
}

inline lapack_int LAPACKE_upgtr(
    char uplo, lapack_int n,
    std::complex<float>* AP,
    std::complex<float>* tau,
    std::complex<float>* Q, lapack_int ldq )
{
    return LAPACKE_cupgtr(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) Q, ldq );
}

inline lapack_int LAPACKE_upgtr(
    char uplo, lapack_int n,
    std::complex<double>* AP,
    std::complex<double>* tau,
    std::complex<double>* Q, lapack_int ldq )
{
    return LAPACKE_zupgtr(
        LAPACK_COL_MAJOR, uplo, n,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) Q, ldq );
}

// -----------------------------------------------------------------------------
inline lapack_int LAPACKE_upmtr(
    char side, char uplo, char trans, lapack_int m, lapack_int n,
    float* AP,
    float* tau,
    float* C, lapack_int ldc )
{
    if (trans == 'C')
        trans = 'T';
    return LAPACKE_sopmtr(
        LAPACK_COL_MAJOR, side, uplo, trans, m, n,
        AP,
        tau,
        C, ldc );
}

inline lapack_int LAPACKE_upmtr(
    char side, char uplo, char trans, lapack_int m, lapack_int n,
    double* AP,
    double* tau,
    double* C, lapack_int ldc )
{
    if (trans == 'C')
        trans = 'T';
    return LAPACKE_dopmtr(
        LAPACK_COL_MAJOR, side, uplo, trans, m, n,
        AP,
        tau,
        C, ldc );
}

inline lapack_int LAPACKE_upmtr(
    char side, char uplo, char trans, lapack_int m, lapack_int n,
    std::complex<float>* AP,
    std::complex<float>* tau,
    std::complex<float>* C, lapack_int ldc )
{
    return LAPACKE_cupmtr(
        LAPACK_COL_MAJOR, side, uplo, trans, m, n,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) C, ldc );
}

inline lapack_int LAPACKE_upmtr(
    char side, char uplo, char trans, lapack_int m, lapack_int n,
    std::complex<double>* AP,
    std::complex<double>* tau,
    std::complex<double>* C, lapack_int ldc )
{
    return LAPACKE_zupmtr(
        LAPACK_COL_MAJOR, side, uplo, trans, m, n,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) C, ldc );
}

#endif // ICL_LAPACKE_WRAPPERS_HH
