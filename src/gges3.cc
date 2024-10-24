// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#if LAPACK_VERSION >= 30600  // >= v3.6

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t gges3(
    lapack::Job jobvsl, lapack::Job jobvsr, lapack::Sort sort,
    lapack_s_select3 select, int64_t n,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    int64_t* sdim,
    std::complex<float>* alpha,
    float* beta,
    float* VSL, int64_t ldvsl,
    float* VSR, int64_t ldvsr )
{
    char jobvsl_ = to_char( jobvsl );
    char jobvsr_ = to_char( jobvsr );
    char sort_ = to_char( sort );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int sdim_ = 0;
    lapack_int ldvsl_ = to_lapack_int( ldvsl );
    lapack_int ldvsr_ = to_lapack_int( ldvsr );
    lapack_int info_ = 0;

    // split-complex representation
    lapack::vector< float > alphar( max( 1, n ) );
    lapack::vector< float > alphai( max( 1, n ) );

    // query for workspace size
    float qry_work[1];
    lapack_int qry_bwork[1];
    lapack_int ineg_one = -1;
    LAPACK_sgges3(
        &jobvsl_, &jobvsr_, &sort_,
        select, &n_,
        A, &lda_,
        B, &ldb_, &sdim_,
        &alphar[0],
        &alphai[0],
        beta,
        VSL, &ldvsl_,
        VSR, &ldvsr_,
        qry_work, &ineg_one,
        qry_bwork, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< float > work( lwork_ );
    lapack::vector< lapack_int > bwork( (n) );

    LAPACK_sgges3(
        &jobvsl_, &jobvsr_, &sort_,
        select, &n_,
        A, &lda_,
        B, &ldb_, &sdim_,
        &alphar[0],
        &alphai[0],
        beta,
        VSL, &ldvsl_,
        VSR, &ldvsr_,
        &work[0], &lwork_,
        &bwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    // merge split-complex representation
    for (int64_t i = 0; i < n; ++i) {
        alpha[i] = std::complex<float>( alphar[i], alphai[i] );
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gges3(
    lapack::Job jobvsl, lapack::Job jobvsr, lapack::Sort sort,
    lapack_d_select3 select, int64_t n,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    int64_t* sdim,
    std::complex<double>* alpha,
    double* beta,
    double* VSL, int64_t ldvsl,
    double* VSR, int64_t ldvsr )
{
    char jobvsl_ = to_char( jobvsl );
    char jobvsr_ = to_char( jobvsr );
    char sort_ = to_char( sort );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int sdim_ = 0;
    lapack_int ldvsl_ = to_lapack_int( ldvsl );
    lapack_int ldvsr_ = to_lapack_int( ldvsr );
    lapack_int info_ = 0;

    // split-complex representation
    lapack::vector< double > alphar( max( 1, n ) );
    lapack::vector< double > alphai( max( 1, n ) );

    // query for workspace size
    double qry_work[1];
    lapack_int qry_bwork[1];
    lapack_int ineg_one = -1;
    LAPACK_dgges3(
        &jobvsl_, &jobvsr_, &sort_,
        select, &n_,
        A, &lda_,
        B, &ldb_, &sdim_,
        &alphar[0],
        &alphai[0],
        beta,
        VSL, &ldvsl_,
        VSR, &ldvsr_,
        qry_work, &ineg_one,
        qry_bwork, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< double > work( lwork_ );
    lapack::vector< lapack_int > bwork( (n) );

    LAPACK_dgges3(
        &jobvsl_, &jobvsr_, &sort_,
        select, &n_,
        A, &lda_,
        B, &ldb_, &sdim_,
        &alphar[0],
        &alphai[0],
        beta,
        VSL, &ldvsl_,
        VSR, &ldvsr_,
        &work[0], &lwork_,
        &bwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    // merge split-complex representation
    for (int64_t i = 0; i < n; ++i) {
        alpha[i] = std::complex<double>( alphar[i], alphai[i] );
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gges3(
    lapack::Job jobvsl, lapack::Job jobvsr, lapack::Sort sort,
    lapack_c_select2 select, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    int64_t* sdim,
    std::complex<float>* alpha,
    std::complex<float>* beta,
    std::complex<float>* VSL, int64_t ldvsl,
    std::complex<float>* VSR, int64_t ldvsr )
{
    char jobvsl_ = to_char( jobvsl );
    char jobvsr_ = to_char( jobvsr );
    char sort_ = to_char( sort );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int sdim_ = 0;
    lapack_int ldvsl_ = to_lapack_int( ldvsl );
    lapack_int ldvsr_ = to_lapack_int( ldvsr );
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    lapack_int qry_bwork[1];
    lapack_int ineg_one = -1;
    LAPACK_cgges3(
        &jobvsl_, &jobvsr_, &sort_,
        (LAPACK_C_SELECT2) select, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_, &sdim_,
        (lapack_complex_float*) alpha,
        (lapack_complex_float*) beta,
        (lapack_complex_float*) VSL, &ldvsl_,
        (lapack_complex_float*) VSR, &ldvsr_,
        (lapack_complex_float*) qry_work, &ineg_one,
        qry_rwork,
        qry_bwork, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );
    lapack::vector< float > rwork( (8*n) );
    lapack::vector< lapack_int > bwork( (n) );

    LAPACK_cgges3(
        &jobvsl_, &jobvsr_, &sort_,
        (LAPACK_C_SELECT2) select, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_, &sdim_,
        (lapack_complex_float*) alpha,
        (lapack_complex_float*) beta,
        (lapack_complex_float*) VSL, &ldvsl_,
        (lapack_complex_float*) VSR, &ldvsr_,
        (lapack_complex_float*) &work[0], &lwork_,
        &rwork[0],
        &bwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gges3(
    lapack::Job jobvsl, lapack::Job jobvsr, lapack::Sort sort,
    lapack_z_select2 select, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    int64_t* sdim,
    std::complex<double>* alpha,
    std::complex<double>* beta,
    std::complex<double>* VSL, int64_t ldvsl,
    std::complex<double>* VSR, int64_t ldvsr )
{
    char jobvsl_ = to_char( jobvsl );
    char jobvsr_ = to_char( jobvsr );
    char sort_ = to_char( sort );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int sdim_ = 0;
    lapack_int ldvsl_ = to_lapack_int( ldvsl );
    lapack_int ldvsr_ = to_lapack_int( ldvsr );
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    lapack_int qry_bwork[1];
    lapack_int ineg_one = -1;
    LAPACK_zgges3(
        &jobvsl_, &jobvsr_, &sort_,
        (LAPACK_Z_SELECT2) select, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_, &sdim_,
        (lapack_complex_double*) alpha,
        (lapack_complex_double*) beta,
        (lapack_complex_double*) VSL, &ldvsl_,
        (lapack_complex_double*) VSR, &ldvsr_,
        (lapack_complex_double*) qry_work, &ineg_one,
        qry_rwork,
        qry_bwork, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );
    lapack::vector< double > rwork( (8*n) );
    lapack::vector< lapack_int > bwork( (n) );

    LAPACK_zgges3(
        &jobvsl_, &jobvsr_, &sort_,
        (LAPACK_Z_SELECT2) select, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_, &sdim_,
        (lapack_complex_double*) alpha,
        (lapack_complex_double*) beta,
        (lapack_complex_double*) VSL, &ldvsl_,
        (lapack_complex_double*) VSR, &ldvsr_,
        (lapack_complex_double*) &work[0], &lwork_,
        &rwork[0],
        &bwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= v3.6
