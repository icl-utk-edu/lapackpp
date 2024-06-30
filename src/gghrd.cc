// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t gghrd(
    lapack::Job compq, lapack::Job compz, int64_t n, int64_t ilo, int64_t ihi,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    float* Q, int64_t ldq,
    float* Z, int64_t ldz )
{
    char compq_ = to_char_comp( compq );
    char compz_ = to_char_comp( compz );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ilo_ = to_lapack_int( ilo );
    lapack_int ihi_ = to_lapack_int( ihi );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    LAPACK_sgghrd(
        &compq_, &compz_, &n_, &ilo_, &ihi_,
        A, &lda_,
        B, &ldb_,
        Q, &ldq_,
        Z, &ldz_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gghrd(
    lapack::Job compq, lapack::Job compz, int64_t n, int64_t ilo, int64_t ihi,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    double* Q, int64_t ldq,
    double* Z, int64_t ldz )
{
    char compq_ = to_char_comp( compq );
    char compz_ = to_char_comp( compz );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ilo_ = to_lapack_int( ilo );
    lapack_int ihi_ = to_lapack_int( ihi );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    LAPACK_dgghrd(
        &compq_, &compz_, &n_, &ilo_, &ihi_,
        A, &lda_,
        B, &ldb_,
        Q, &ldq_,
        Z, &ldz_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gghrd(
    lapack::Job compq, lapack::Job compz, int64_t n, int64_t ilo, int64_t ihi,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* Q, int64_t ldq,
    std::complex<float>* Z, int64_t ldz )
{
    char compq_ = to_char_comp( compq );
    char compz_ = to_char_comp( compz );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ilo_ = to_lapack_int( ilo );
    lapack_int ihi_ = to_lapack_int( ihi );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    LAPACK_cgghrd(
        &compq_, &compz_, &n_, &ilo_, &ihi_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) Q, &ldq_,
        (lapack_complex_float*) Z, &ldz_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gghrd(
    lapack::Job compq, lapack::Job compz, int64_t n, int64_t ilo, int64_t ihi,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* Q, int64_t ldq,
    std::complex<double>* Z, int64_t ldz )
{
    char compq_ = to_char_comp( compq );
    char compz_ = to_char_comp( compz );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ilo_ = to_lapack_int( ilo );
    lapack_int ihi_ = to_lapack_int( ihi );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    LAPACK_zgghrd(
        &compq_, &compz_, &n_, &ilo_, &ihi_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) Q, &ldq_,
        (lapack_complex_double*) Z, &ldz_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
