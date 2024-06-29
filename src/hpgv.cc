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
int64_t hpgv(
    int64_t itype, lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    std::complex<float>* AP,
    std::complex<float>* BP,
    float* W,
    std::complex<float>* Z, int64_t ldz )
{
    lapack_int itype_ = to_lapack_int( itype );
    char jobz_ = to_char( jobz );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (max( 1, 2*n-1 )) );
    lapack::vector< float > rwork( (max( 1, 3*n-2 )) );

    LAPACK_chpgv(
        &itype_, &jobz_, &uplo_, &n_,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) BP,
        W,
        (lapack_complex_float*) Z, &ldz_,
        (lapack_complex_float*) &work[0],
        &rwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t hpgv(
    int64_t itype, lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    std::complex<double>* AP,
    std::complex<double>* BP,
    double* W,
    std::complex<double>* Z, int64_t ldz )
{
    lapack_int itype_ = to_lapack_int( itype );
    char jobz_ = to_char( jobz );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (max( 1, 2*n-1 )) );
    lapack::vector< double > rwork( (max( 1, 3*n-2 )) );

    LAPACK_zhpgv(
        &itype_, &jobz_, &uplo_, &n_,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) BP,
        W,
        (lapack_complex_double*) Z, &ldz_,
        (lapack_complex_double*) &work[0],
        &rwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
