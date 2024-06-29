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
int64_t ggbal(
    lapack::Balance balance, int64_t n,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    int64_t* ilo,
    int64_t* ihi,
    float* lscale,
    float* rscale )
{
    char balance_ = to_char( balance );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ilo_ = to_lapack_int( *ilo );
    lapack_int ihi_ = to_lapack_int( *ihi );
    lapack_int info_ = 0;

    // from docs
    int64_t lwork = (balance == Balance::Scale || balance == Balance::Both ? max( 1, 6*n ) : 1);

    // allocate workspace
    lapack::vector< float > work( (lwork) );

    LAPACK_sggbal(
        &balance_, &n_,
        A, &lda_,
        B, &ldb_, &ilo_, &ihi_,
        lscale,
        rscale,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbal(
    lapack::Balance balance, int64_t n,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    int64_t* ilo,
    int64_t* ihi,
    double* lscale,
    double* rscale )
{
    char balance_ = to_char( balance );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ilo_ = to_lapack_int( *ilo );
    lapack_int ihi_ = to_lapack_int( *ihi );
    lapack_int info_ = 0;

    // from docs
    int64_t lwork = (balance == Balance::Scale || balance == Balance::Both ? max( 1, 6*n ) : 1);

    // allocate workspace
    lapack::vector< double > work( (lwork) );

    LAPACK_dggbal(
        &balance_, &n_,
        A, &lda_,
        B, &ldb_, &ilo_, &ihi_,
        lscale,
        rscale,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbal(
    lapack::Balance balance, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    int64_t* ilo,
    int64_t* ihi,
    float* lscale,
    float* rscale )
{
    char balance_ = to_char( balance );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ilo_ = to_lapack_int( *ilo );
    lapack_int ihi_ = to_lapack_int( *ihi );
    lapack_int info_ = 0;

    // from docs
    int64_t lwork = (balance == Balance::Scale || balance == Balance::Both ? max( 1, 6*n ) : 1);

    // allocate workspace
    lapack::vector< float > work( (lwork) );

    LAPACK_cggbal(
        &balance_, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_, &ilo_, &ihi_,
        lscale,
        rscale,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbal(
    lapack::Balance balance, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    int64_t* ilo,
    int64_t* ihi,
    double* lscale,
    double* rscale )
{
    char balance_ = to_char( balance );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ilo_ = to_lapack_int( *ilo );
    lapack_int ihi_ = to_lapack_int( *ihi );
    lapack_int info_ = 0;

    // from docs
    int64_t lwork = (balance == Balance::Scale || balance == Balance::Both ? max( 1, 6*n ) : 1);

    // allocate workspace
    lapack::vector< double > work( (lwork) );

    LAPACK_zggbal(
        &balance_, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_, &ilo_, &ihi_,
        lscale,
        rscale,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

}  // namespace lapack
