// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef CBLAS_HH
#define CBLAS_HH

#include "blas/defines.h"

#if defined(BLAS_HAVE_MKL)
    #if defined(BLAS_ILP64) && ! defined(MKL_ILP64)
        #define MKL_ILP64
    #endif
    #include <mkl_cblas.h>

#elif defined(BLAS_HAVE_ESSL)
    #if defined(BLAS_ILP64) && ! defined(_ESV6464)
        #define _ESV6464
    #endif
    #include <essl.h>

#elif defined(BLAS_HAVE_ACCELERATE)
    // On macOS, the official way to include cblas is via Accelerate.h.
    // Unfortunately with Xcode 10.3 and GNU g++ 9.3, that doesn't compile.
    // If we can find cblas.h, use it, otherwise use Accelerate.h.
    #ifdef BLAS_HAVE_ACCELERATE_CBLAS_H
        #include <cblas.h>
    #else
        #include <Accelerate/Accelerate.h>
    #endif
    typedef CBLAS_ORDER CBLAS_LAYOUT;

#else
    // Some ancient cblas.h don't include extern C. It's okay to nest.
    extern "C" {
    #include <cblas.h>
    }

    // Original cblas.h used CBLAS_ORDER; new uses CBLAS_LAYOUT and makes
    // CBLAS_ORDER a typedef. Make sure CBLAS_LAYOUT is defined.
    typedef CBLAS_ORDER CBLAS_LAYOUT;
#endif

#include "blas/util.hh"

#include <complex>

// =============================================================================
// constants

// -----------------------------------------------------------------------------
inline CBLAS_LAYOUT cblas_layout_const( blas::Layout layout )
{
    switch (layout) {
        case blas::Layout::RowMajor:  return CblasRowMajor;
        case blas::Layout::ColMajor:  return CblasColMajor;
        default: throw blas::Error();
    }
}

inline CBLAS_LAYOUT cblas_layout_const( char layout )
{
    switch (layout) {
        case 'r': case 'R': return CblasRowMajor;
        case 'c': case 'C': return CblasColMajor;
        default: throw blas::Error();
    }
}

inline char lapack_layout_const( CBLAS_LAYOUT layout )
{
    switch (layout) {
        case CblasRowMajor: return 'r';
        case CblasColMajor: return 'c';
        default: throw blas::Error();
    }
}


// -----------------------------------------------------------------------------
inline CBLAS_DIAG cblas_diag_const( blas::Diag diag )
{
    switch (diag) {
        case blas::Diag::NonUnit:  return CblasNonUnit;
        case blas::Diag::Unit:     return CblasUnit;
        default: throw blas::Error();
    }
}

inline CBLAS_DIAG cblas_diag_const( char diag )
{
    switch (diag) {
        case 'n': case 'N': return CblasNonUnit;
        case 'u': case 'U': return CblasUnit;
        default: throw blas::Error();
    }
}

inline char lapack_diag_const( CBLAS_DIAG diag )
{
    switch (diag) {
        case CblasNonUnit: return 'n';
        case CblasUnit: return 'u';
        default: throw blas::Error();
    }
}


// -----------------------------------------------------------------------------
inline CBLAS_SIDE cblas_side_const( blas::Side side )
{
    switch (side) {
        case blas::Side::Left:  return CblasLeft;
        case blas::Side::Right: return CblasRight;
        default: throw blas::Error();
    }
}

inline CBLAS_SIDE cblas_side_const( char side )
{
    switch (side) {
        case 'l': case 'L': return CblasLeft;
        case 'r': case 'R': return CblasRight;
        default: throw blas::Error();
    }
}

inline char lapack_side_const( CBLAS_SIDE side )
{
    switch (side) {
        case CblasLeft:  return 'l';
        case CblasRight: return 'r';
        default: throw blas::Error();
    }
}


// -----------------------------------------------------------------------------
inline CBLAS_TRANSPOSE cblas_trans_const( blas::Op trans )
{
    switch (trans) {
        case blas::Op::NoTrans:   return CblasNoTrans;
        case blas::Op::Trans:     return CblasTrans;
        case blas::Op::ConjTrans: return CblasConjTrans;
        default: throw blas::Error();
    }
}

inline CBLAS_TRANSPOSE cblas_trans_const( char trans )
{
    switch (trans) {
        case 'n': case 'N': return CblasNoTrans;
        case 't': case 'T': return CblasTrans;
        case 'c': case 'C': return CblasConjTrans;
        default: throw blas::Error();
    }
}

inline char lapack_trans_const( CBLAS_TRANSPOSE trans )
{
    switch (trans) {
        case CblasNoTrans:   return 'n';
        case CblasTrans:     return 't';
        case CblasConjTrans: return 'c';
        default: throw blas::Error();
    }
}


// -----------------------------------------------------------------------------
inline CBLAS_UPLO cblas_uplo_const( blas::Uplo uplo )
{
    switch (uplo) {
        case blas::Uplo::Lower: return CblasLower;
        case blas::Uplo::Upper: return CblasUpper;
        default: throw blas::Error();
    }
}

inline CBLAS_UPLO cblas_uplo_const( char uplo )
{
    switch (uplo) {
        case 'l': case 'L': return CblasLower;
        case 'u': case 'U': return CblasUpper;
        default: throw blas::Error();
    }
}

inline char lapack_uplo_const( CBLAS_UPLO uplo )
{
    switch (uplo) {
        case CblasLower: return 'l';
        case CblasUpper: return 'u';
        default: throw blas::Error();
    }
}


// =============================================================================
// Level 2 BLAS

// -----------------------------------------------------------------------------
inline void
cblas_symv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    float  alpha,
    float const *A, int lda,
    float const *x, int incx,
    float  beta,
    float* y, int incy )
{
    cblas_ssymv( layout, uplo, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_symv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    double  alpha,
    double const *A, int lda,
    double const *x, int incx,
    double  beta,
    double* y, int incy )
{
    cblas_dsymv( layout, uplo, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

// LAPACK provides [cz]symv, CBLAS lacks them
void
cblas_symv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    int n,
    std::complex<float> alpha,
    std::complex<float> const* A, int lda,
    std::complex<float> const* x, int incx,
    std::complex<float> beta,
    std::complex<float>* yref, int incy );

void
cblas_symv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    int n,
    std::complex<double> alpha,
    std::complex<double> const* A, int lda,
    std::complex<double> const* x, int incx,
    std::complex<double> beta,
    std::complex<double>* yref, int incy );

// -----------------------------------------------------------------------------
inline void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    float alpha,
    float const *x, int incx,
    float* A, int lda )
{
    cblas_ssyr( layout, uplo, n, alpha, x, incx, A, lda );
}

inline void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    double alpha,
    double const *x, int incx,
    double* A, int lda )
{
    cblas_dsyr( layout, uplo, n, alpha, x, incx, A, lda );
}

// LAPACK provides [cz]syr, CBLAS lacks them
void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float>* A, int lda );

void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double>* A, int lda );

// -----------------------------------------------------------------------------
inline void
cblas_gbmv(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans,
    int m, int n, int kl, int ku,
    float alpha,
    float const *A, int lda,
    float const *x, int incx,
    float beta,
    float *y, int incy )
{
    cblas_sgbmv( layout, trans, m, n, kl, ku, alpha, A, lda,
                 x, incx, beta, y, incy );
}

inline void
cblas_gbmv(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans,
    int m, int n, int kl, int ku,
    double alpha,
    double const *A, int lda,
    double const *x, int incx,
    double beta,
    double *y, int incy )
{
    cblas_dgbmv( layout, trans, m, n, kl, ku, alpha, A, lda,
                 x, incx, beta, y, incy );
}

inline void
cblas_gbmv(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans,
    int m, int n, int kl, int ku,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *x, int incx,
    std::complex<float> beta,
    std::complex<float> *y, int incy )
{
    cblas_cgbmv( layout, trans, m, n, kl, ku, &alpha, A, lda,
                 x, incx, &beta, y, incy );
}

inline void
cblas_gbmv(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans,
    int m, int n, int kl, int ku,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *x, int incx,
    std::complex<double> beta,
    std::complex<double> *y, int incy )
{
    cblas_zgbmv( layout, trans, m, n, kl, ku, &alpha, A, lda,
                 x, incx, &beta, y, incy );
}

// -----------------------------------------------------------------------------
inline void
cblas_hbmv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo,
    int n, int kd,
    float alpha,
    float const *A, int lda,
    float const *x, int incx,
    float beta,
    float *y, int incy )
{
    cblas_ssbmv( layout, uplo, n, kd, alpha, A, lda,
                 x, incx, beta, y, incy );
}

inline void
cblas_hbmv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo,
    int n, int kd,
    double alpha,
    double const *A, int lda,
    double const *x, int incx,
    double beta,
    double *y, int incy )
{
    cblas_dsbmv( layout, uplo, n, kd, alpha, A, lda,
                 x, incx, beta, y, incy );
}

inline void
cblas_hbmv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo,
    int n, int kd,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *x, int incx,
    std::complex<float> beta,
    std::complex<float> *y, int incy )
{
    cblas_chbmv( layout, uplo, n, kd, &alpha, A, lda,
                 x, incx, &beta, y, incy );
}

inline void
cblas_hbmv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo,
    int n, int kd,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *x, int incx,
    std::complex<double> beta,
    std::complex<double> *y, int incy )
{
    cblas_zhbmv( layout, uplo, n, kd, &alpha, A, lda,
                 x, incx, &beta, y, incy );
}

#endif        //  #ifndef CBLAS_HH
