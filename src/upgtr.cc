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
int64_t upgtr(
    lapack::Uplo uplo, int64_t n,
    std::complex<float> const* AP,
    std::complex<float> const* tau,
    std::complex<float>* Q, int64_t ldq )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (n-1) );

    LAPACK_cupgtr(
        &uplo_, &n_,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) Q, &ldq_,
        (lapack_complex_float*) &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t upgtr(
    lapack::Uplo uplo, int64_t n,
    std::complex<double> const* AP,
    std::complex<double> const* tau,
    std::complex<double>* Q, int64_t ldq )
{
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (n-1) );

    LAPACK_zupgtr(
        &uplo_, &n_,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) Q, &ldq_,
        (lapack_complex_double*) &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
