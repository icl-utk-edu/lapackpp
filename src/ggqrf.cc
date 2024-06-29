// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t ggqrf(
    int64_t n, int64_t m, int64_t p,
    float* A, int64_t lda,
    float* taua,
    float* B, int64_t ldb,
    float* taub )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int m_ = to_lapack_int( m );
    lapack_int p_ = to_lapack_int( p );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_sggqrf(
        &n_, &m_, &p_,
        A, &lda_,
        taua,
        B, &ldb_,
        taub,
        qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< float > work( lwork_ );

    LAPACK_sggqrf(
        &n_, &m_, &p_,
        A, &lda_,
        taua,
        B, &ldb_,
        taub,
        &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggqrf(
    int64_t n, int64_t m, int64_t p,
    double* A, int64_t lda,
    double* taua,
    double* B, int64_t ldb,
    double* taub )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int m_ = to_lapack_int( m );
    lapack_int p_ = to_lapack_int( p );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_dggqrf(
        &n_, &m_, &p_,
        A, &lda_,
        taua,
        B, &ldb_,
        taub,
        qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< double > work( lwork_ );

    LAPACK_dggqrf(
        &n_, &m_, &p_,
        A, &lda_,
        taua,
        B, &ldb_,
        taub,
        &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggqrf(
    int64_t n, int64_t m, int64_t p,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* taua,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* taub )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int m_ = to_lapack_int( m );
    lapack_int p_ = to_lapack_int( p );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_cggqrf(
        &n_, &m_, &p_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) taua,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) taub,
        (lapack_complex_float*) qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );

    LAPACK_cggqrf(
        &n_, &m_, &p_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) taua,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) taub,
        (lapack_complex_float*) &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggqrf(
    int64_t n, int64_t m, int64_t p,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* taua,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* taub )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int m_ = to_lapack_int( m );
    lapack_int p_ = to_lapack_int( p );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_zggqrf(
        &n_, &m_, &p_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) taua,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) taub,
        (lapack_complex_double*) qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );

    LAPACK_zggqrf(
        &n_, &m_, &p_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) taua,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) taub,
        (lapack_complex_double*) &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
