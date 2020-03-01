// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_FLOPS_HH
#define LAPACK_FLOPS_HH

#include "lapack.hh"
#include "blas/flops.hh"

#include <complex>

namespace lapack {

//==============================================================================
// Generic formulas come from LAWN 41
// BLAS formulas generally assume alpha == 1 or -1, and beta == 1, -1, or 0;
// otherwise add some smaller order term.
// Some formulas are wrong when m, n, or k == 0; flops should be 0
// (e.g., syr2k, unmqr).
// Formulas may give negative results for invalid combinations of m, n, k
// (e.g., ungqr, unmqr).

//------------------------------------------------------------ getrf
// LAWN 41 omits (m < n) case
inline double fmuls_getrf(double m, double n)
{
    return (m >= n)
        ? (0.5*m*n*n - 1./6*n*n*n + 0.5*m*n - 0.5*n*n + 2/3.*n)
        : (0.5*n*m*m - 1./6*m*m*m + 0.5*n*m - 0.5*m*m + 2/3.*m);
}

inline double fadds_getrf(double m, double n)
{
    return (m >= n)
        ? (0.5*m*n*n - 1./6*n*n*n - 0.5*m*n + 1./6*n)
        : (0.5*n*m*m - 1./6*m*m*m - 0.5*n*m + 1./6*m);
}

//------------------------------------------------------------ getri
inline double fmuls_getri(double n)
    { return 2/3.*n*n*n + 0.5*n*n + 5./6*n; }

inline double fadds_getri(double n)
    { return 2/3.*n*n*n - 1.5*n*n + 5./6*n; }

//------------------------------------------------------------ getrs
inline double fmuls_getrs(double n, double nrhs)
    { return nrhs*n*n; }

inline double fadds_getrs(double n, double nrhs)
    { return nrhs*n*(n - 1); }

//------------------------------------------------------------ potrf
inline double fmuls_potrf(double n)
    { return 1./6*n*n*n + 0.5*n*n + 1./3.*n; }

inline double fadds_potrf(double n)
    { return 1./6*n*n*n - 1./6*n; }

//------------------------------------------------------------ potri
inline double fmuls_potri(double n)
    { return 1./3.*n*n*n + n*n + 2/3.*n; }

inline double fadds_potri(double n)
    { return 1./3.*n*n*n - 0.5*n*n + 1./6*n; }

//------------------------------------------------------------ potrs
inline double fmuls_potrs(double n, double nrhs)
    { return nrhs*n*(n + 1); }

inline double fadds_potrs(double n, double nrhs)
    { return nrhs*n*(n - 1); }

//------------------------------------------------------------ pbtrf
inline double fmuls_pbtrf(double n, double k)
    { return n*(1./2.*k*k + 3./2.*k + 1) - 1./3.*k*k*k - k*k - 2./3.*k; }

inline double fadds_pbtrf(double n, double k)
    { return n*(1./2.*k*k + 1./2.*k) - 1./3.*k*k*k - 1./2.*k*k - 1./6.*k; }

//------------------------------------------------------------ pbtrs
inline double fmuls_pbtrs(double n, double nrhs, double k)
    { return nrhs*(2*n*k + 2*n - k*k - k); }

inline double fadds_pbtrs(double n, double nrhs, double k)
    { return nrhs*(2*n*k - k*k - k); }

//------------------------------------------------------------ sytrf
inline double fmuls_sytrf(double n)
    { return 1/6.*n*n*n + 0.5*n*n + 10/3.*n; }

inline double fadds_sytrf(double n)
    { return 1/6.*n*n*n - 1/6.*n; }

//------------------------------------------------------------ sytri
inline double fmuls_sytri(double n)
    { return 1/3.*n*n*n + n*n + 2/3.*n; }

inline double fadds_sytri(double n)
    { return 1/3.*n*n*n - 1/3.*n; }

//------------------------------------------------------------ sytrs
inline double fmuls_sytrs(double n, double nrhs)
    { return nrhs*n*(n + 1); }

inline double fadds_sytrs(double n, double nrhs)
    { return nrhs*n*(n - 1); }

//------------------------------------------------------------ geqrf
inline double fmuls_geqrf(double m, double n)
{
    return (m > n)
        ? (m*n*n - 1./3.*n*n*n +   m*n + 0.5*n*n + 23./6*n)
        : (n*m*m - 1./3.*m*m*m + 2*n*m - 0.5*m*m + 23./6*m);
}

inline double fadds_geqrf(double m, double n)
{
    return (m > n)
        ? (m*n*n - 1./3.*n*n*n + 0.5*n*n       + 5./6*n)
        : (n*m*m - 1./3.*m*m*m + n*m - 0.5*m*m + 5./6*m);
}

//------------------------------------------------------------ geqrt
// TODO: this seems odd -- should it match geqrf? At least be O(mn^2)?
inline double fmuls_geqrt(double m, double n)
    { return 0.5*m*n; }

inline double fadds_geqrt(double m, double n)
    { return 0.5*m*n; }

//------------------------------------------------------------ geqlf
inline double fmuls_geqlf(double m, double n)
    { return fmuls_geqrf(m, n); }

inline double fadds_geqlf(double m, double n)
    { return fadds_geqrf(m, n); }

//------------------------------------------------------------ gerqf
inline double fmuls_gerqf(double m, double n)
{
    return (m > n)
        ? (m*n*n - 1./3.*n*n*n +   m*n + 0.5*n*n + 29./6*n)
        : (n*m*m - 1./3.*m*m*m + 2*n*m - 0.5*m*m + 29./6*m);
}

inline double fadds_gerqf(double m, double n)
{
    return (m > n)
        ? (m*n*n - 1./3.*n*n*n + m*n - 0.5*n*n + 5./6*n)
        : (n*m*m - 1./3.*m*m*m + 0.5*m*m       + 5./6*m);
}

//------------------------------------------------------------ gelqf
inline double fmuls_gelqf(double m, double n)
    { return  fmuls_gerqf(m, n); }

inline double fadds_gelqf(double m, double n)
    { return  fadds_gerqf(m, n); }

//------------------------------------------------------------ ungqr
inline double fmuls_ungqr(double m, double n, double k)
    { return 2*m*n*k - (m + n)*k*k + 2/3.*k*k*k + 2*n*k - k*k - 5./3.*k; }

inline double fadds_ungqr(double m, double n, double k)
    { return 2*m*n*k - (m + n)*k*k + 2/3.*k*k*k + n*k - m*k + 1./3.*k; }

//------------------------------------------------------------ ungql
inline double fmuls_ungql(double m, double n, double k)
    { return  fmuls_ungqr(m, n, k); }

inline double fadds_ungql(double m, double n, double k)
    { return fadds_ungqr(m, n, k); }

//------------------------------------------------------------ ungrq
inline double fmuls_ungrq(double m, double n, double k)
    { return 2*m*n*k - (m + n)*k*k + 2/3.*k*k*k + m*k + n*k - k*k - 2/3.*k; }

inline double fadds_ungrq(double m, double n, double k)
    { return 2*m*n*k - (m + n)*k*k + 2/3.*k*k*k + m*k - n*k + 1./3.*k; }

//------------------------------------------------------------ unglq
inline double fmuls_unglq(double m, double n, double k)
    { return fmuls_ungrq(m, n, k); }

inline double fadds_unglq(double m, double n, double k)
    { return fadds_ungrq(m, n, k); }

//------------------------------------------------------------ unmqr
inline double fmuls_unmqr(lapack::Side side, double m, double n, double k)
{
    return (side == lapack::Side::Left)
        ? (2*n*m*k - n*k*k + 2*n*k)
        : (2*n*m*k - m*k*k + m*k + n*k - 0.5*k*k + 0.5*k);
}

inline double fadds_unmqr(lapack::Side side, double m, double n, double k)
{
    return (side == lapack::Side::Left)
        ? (2*n*m*k - n*k*k + n*k)
        : (2*n*m*k - m*k*k + m*k);
}

//------------------------------------------------------------ unmql
inline double fmuls_unmql(lapack::Side side, double m, double n, double k)
    { return fmuls_unmqr(side, m, n, k); }

inline double fadds_unmql(lapack::Side side, double m, double n, double k)
    { return fadds_unmqr(side, m, n, k); }

//------------------------------------------------------------ unmrq
inline double fmuls_unmrq(lapack::Side side, double m, double n, double k)
    { return fmuls_unmqr(side, m, n, k); }

inline double fadds_unmrq(lapack::Side side, double m, double n, double k)
    { return fadds_unmqr(side, m, n, k); }

//------------------------------------------------------------ unmlq
inline double fmuls_unmlq(lapack::Side side, double m, double n, double k)
    { return fmuls_unmqr(side, m, n, k); }

inline double fadds_unmlq(lapack::Side side, double m, double n, double k)
    { return fadds_unmqr(side, m, n, k); }

//------------------------------------------------------------ trtri
inline double fmuls_trtri(double n)
    { return 1./6*n*n*n + 0.5*n*n + 1./3.*n; }

inline double fadds_trtri(double n)
    { return 1./6*n*n*n - 0.5*n*n + 1./3.*n; }

//------------------------------------------------------------ gehrd
inline double fmuls_gehrd(double n)
    { return 5./3.*n*n*n + 0.5*n*n - 7./6*n; }

inline double fadds_gehrd(double n)
    { return 5./3.*n*n*n - n*n - 2/3.*n; }

//------------------------------------------------------------ sytrd
inline double fmuls_sytrd(double n)
    { return 2/3.*n*n*n + 2.5*n*n - 1./6*n; }

inline double fadds_sytrd(double n)
    { return 2/3.*n*n*n + n*n - 8./3.*n; }

inline double fmuls_hetrd(double n)
    { return fmuls_sytrd(n); }

inline double fadds_hetrd(double n)
    { return fadds_sytrd(n); }

//------------------------------------------------------------ gebrd
inline double fmuls_gebrd(double m, double n)
{
    return (m >= n)
        ? (2*m*n*n - 2/3.*n*n*n + 2*n*n + 20./3.*n)
        : (2*n*m*m - 2/3.*m*m*m + 2*m*m + 20./3.*m);
}

inline double fadds_gebrd(double m, double n)
{
    return (m >= n)
        ? (2*m*n*n - 2/3.*n*n*n + n*n - m*n +  5./3.*n)
        : (2*n*m*m - 2/3.*m*m*m + m*m - n*m +  5./3.*m);
}

//------------------------------------------------------------ larfg
inline double fmuls_larfg(double n)
    { return 2*n; }

inline double fadds_larfg(double n)
    { return   n; }

//------------------------------------------------------------ geadd
inline double fmuls_geadd(double m, double n)
    { return 2*m*n; }

inline double fadds_geadd(double m, double n)
    { return   m*n; }

//------------------------------------------------------------ lauum
inline double fmuls_lauum(double n)
    { return fmuls_potri(n) - fmuls_trtri(n); }

inline double fadds_lauum(double n)
    { return fadds_potri(n) - fadds_trtri(n); }

//------------------------------------------------------------ lange
inline double fmuls_lange(lapack::Norm norm, double m, double n)
    { return norm == lapack::Norm::Fro ? m*n : 0; }

inline double fadds_lange(lapack::Norm norm, double m, double n)
{
    switch (norm) {
    case lapack::Norm::One: return (m-1)*n;
    case lapack::Norm::Inf: return (n-1)*m;
    case lapack::Norm::Fro: return m*n-1;
    default:                return 0;
    }
}

//------------------------------------------------------------ lanhe
inline double fmuls_lanhe(lapack::Norm norm, double n)
    { return norm == lapack::Norm::Fro ? n*(n+1)/2 : 0; }

inline double fadds_lanhe(lapack::Norm norm, double n)
{
    switch (norm) {
    case lapack::Norm::One: return (n-1)*n;
    case lapack::Norm::Inf: return (n-1)*n;
    case lapack::Norm::Fro: return n*(n+1)/2-1;
    default:                return 0;
    }
}

//==============================================================================
// template class. Example:
// gbyte< float >::gemv( m, n ) yields bytes transferred for sgemv.
// gbyte< std::complex<float> >::gemv( m, n ) yields bytes transferred for cgemv.
//==============================================================================
template< typename T >
class Gbyte:
    public blas::Gbyte<T>
{
};

//==============================================================================
// template class. Example:
// gflop< float >::getrf( m, n ) yields flops for sgetrf.
// gflop< std::complex<float> >::getrf( m, n ) yields flops for cgetrf.
//==============================================================================
template< typename T >
class Gflop:
    public blas::Gflop<T>
{
public:
    using blas::Gflop<T>::mul_ops;
    using blas::Gflop<T>::add_ops;

    // LU
    static double gesv(double n, double nrhs)
        { return getrf(n, n) + getrs(n, nrhs); }

    static double getrf(double m, double n)
        { return 1e-9 * (mul_ops*fmuls_getrf(m, n) + add_ops*fadds_getrf(m, n)); }

    static double getri(double n)
        { return 1e-9 * (mul_ops*fmuls_getri(n) + add_ops*fadds_getri(n)); }

    static double getrs(double n, double nrhs)
        { return 1e-9 * (mul_ops*fmuls_getrs(n, nrhs) + add_ops*fadds_getrs(n, nrhs)); }

    // Cholesky
    static double posv(double n, double nrhs)
        { return potrf(n) + potrs(n, nrhs); }

    static double potrf(double n)
        { return 1e-9 * (mul_ops*fmuls_potrf(n) + add_ops*fadds_potrf(n)); }

    static double potri(double n)
        { return 1e-9 * (mul_ops*fmuls_potri(n) + add_ops*fadds_potri(n)); }

    static double potrs(double n, double nrhs)
        { return 1e-9 * (mul_ops*fmuls_potrs(n, nrhs) + add_ops*fadds_potrs(n, nrhs)); }

    // Band Cholesky
    static double pbsv(double n, double nrhs, double k)
        { return pbtrf(n, k) + pbtrs(n, nrhs, k); }

    static double pbtrf(double n, double k)
        { return 1e-9 * (mul_ops*fmuls_pbtrf(n, k) + add_ops*fadds_pbtrf(n, k)); }

    static double pbtrs(double n, double nrhs, double k)
        { return 1e-9 * (mul_ops*fmuls_pbtrs(n, nrhs, k) + add_ops*fadds_pbtrs(n, nrhs, k)); }

    // LDL^T
    static double sysv(double n, double nrhs)
        { return sytrf(n) + sytrs(n, nrhs); }

    static double sytrf(double n)
        { return 1e-9 * (mul_ops*fmuls_sytrf(n) + add_ops*fadds_sytrf(n)); }

    static double sytri(double n)
        { return 1e-9 * (mul_ops*fmuls_sytri(n) + add_ops*fadds_sytri(n)); }

    static double sytrs(double n, double nrhs)
        { return 1e-9 * (mul_ops*fmuls_sytrs(n, nrhs) + add_ops*fadds_sytrs(n, nrhs)); }

    static double hesv(double n, double nrhs)
        { return sysv(n, nrhs); }

    static double hetrf(double n)
        { return sytrf(n); }

    static double hetri(double n)
        { return sytri(n); }

    static double hetrs(double n, double nrhs)
        { return sytrs(n, nrhs); }

    // QR, QL, RQ, LQ
    static double geqrf(double m, double n)
        { return 1e-9 * (mul_ops*fmuls_geqrf(m, n) + add_ops*fadds_geqrf(m, n)); }

    static double geqrt(double m, double n)
        { return 1e-9 * (mul_ops*fmuls_geqrt(m, n) + add_ops*fadds_geqrt(m, n)); }

    static double geqlf(double m, double n)
        { return 1e-9 * (mul_ops*fmuls_geqlf(m, n) + add_ops*fadds_geqlf(m, n)); }

    static double gerqf(double m, double n)
        { return 1e-9 * (mul_ops*fmuls_gerqf(m, n) + add_ops*fadds_gerqf(m, n)); }

    static double gelqf(double m, double n)
        { return 1e-9 * (mul_ops*fmuls_gelqf(m, n) + add_ops*fadds_gelqf(m, n)); }

    // generate Q
    static double ungqr(double m, double n, double k)
        { return 1e-9 * (mul_ops*fmuls_ungqr(m, n, k) + add_ops*fadds_ungqr(m, n, k)); }

    static double orgqr(double m, double n, double k)
        { return ungqr(m, n, k); }

    static double ungql(double m, double n, double k)
        { return 1e-9 * (mul_ops*fmuls_ungql(m, n, k) + add_ops*fadds_ungql(m, n, k)); }

    static double orgql(double m, double n, double k)
        { return ungql(m, n, k); }

    static double ungrq(double m, double n, double k)
        { return 1e-9 * (mul_ops*fmuls_ungrq(m, n, k) + add_ops*fadds_ungrq(m, n, k)); }

    static double orgrq(double m, double n, double k)
        { return ungrq(m, n, k); }

    static double unglq(double m, double n, double k)
        { return 1e-9 * (mul_ops*fmuls_unglq(m, n, k) + add_ops*fadds_unglq(m, n, k)); }

    static double orglq(double m, double n, double k)
        { return unglq(m, n, k); }

    // multiply by Q
    static double unmqr(lapack::Side side, double m, double n, double k)
        { return 1e-9 * (mul_ops*fmuls_unmqr(side, m, n, k) + add_ops*fadds_unmqr(side, m, n, k)); }

    static double ormqr(lapack::Side side, double m, double n, double k)
        { return unmqr(side, m, n, k); }

    static double unmql(lapack::Side side, double m, double n, double k)
        { return 1e-9 * (mul_ops*fmuls_unmql(side, m, n, k) + add_ops*fadds_unmql(side, m, n, k)); }

    static double ormql(lapack::Side side, double m, double n, double k)
        { return unmql(side, m, n, k); }

    static double unmrq(lapack::Side side, double m, double n, double k)
        { return 1e-9 * (mul_ops*fmuls_unmrq(side, m, n, k) + add_ops*fadds_unmrq(side, m, n, k)); }

    static double ormrq(lapack::Side side, double m, double n, double k)
        { return unmrq(side, m, n, k); }

    static double unmlq(lapack::Side side, double m, double n, double k)
        { return 1e-9 * (mul_ops*fmuls_unmlq(side, m, n, k) + add_ops*fadds_unmlq(side, m, n, k)); }

    static double ormlq(lapack::Side side, double m, double n, double k)
        { return unmlq(side, m, n, k); }

    // least squares
    static double gels(double m, double n, double nrhs)
    {
        blas::Side left = blas::Side::Left;
        return (m >= n
            ? geqrf(m, n) + unmqr(left, m, nrhs, n) + blas::Gflop<T>::trsm(left, n, nrhs)
            : gelqf(m, n) + unmlq(left, n, nrhs, m) + blas::Gflop<T>::trsm(left, m, nrhs));
    }

    // triangle inverse
    static double trtri(double n)
        { return 1e-9 * (mul_ops*fmuls_trtri(n) + add_ops*fadds_trtri(n)); }

    // Hessenberg reduction (non-symmetric eigenvalue)
    static double gehrd(double n)
        { return 1e-9 * (mul_ops*fmuls_gehrd(n) + add_ops*fadds_gehrd(n)); }

    // tridiagonal reduction (symmetric eigenvalue)
    static double hetrd(double n)
        { return 1e-9 * (mul_ops*fmuls_sytrd(n) + add_ops*fadds_sytrd(n)); }

    static double sytrd(double n)
        { return hetrd(n); }

    // bidiagonal reduction (SVD)
    static double gebrd(double m, double n)
        { return 1e-9 * (mul_ops*fmuls_gebrd(m, n) + add_ops*fadds_gebrd(m, n)); }

    // Householder reflector generate
    static double larfg(double n)
        { return 1e-9 * (mul_ops*fmuls_larfg(n) + add_ops*fadds_larfg(n)); }

    // matrix add
    static double geadd(double m, double n)
        { return 1e-9 * (mul_ops*fmuls_geadd(m, n) + add_ops*fadds_geadd(m, n)); }

    // U^H*U or L*L^T
    static double lauum(double n)
        { return 1e-9 * (mul_ops*fmuls_lauum(n) + add_ops*fadds_lauum(n)); }

    // norm
    static double lange(lapack::Norm norm, double m, double n)
        { return 1e-9 * (mul_ops*fmuls_lange(norm, m, n) + add_ops*fadds_lange(norm, m, n)); }

    static double lanhe(lapack::Norm norm, double n)
        { return 1e-9 * (mul_ops*fmuls_lanhe(norm, n) + add_ops*fadds_lanhe(norm, n)); }

    static double lansy(lapack::Norm norm, double n)
        { return lanhe(norm, n); }
};

}  // namespace lapack

#endif  // LAPACK_FLOPS_HH
