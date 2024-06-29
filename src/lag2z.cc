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
int64_t lag2z(
    int64_t m, int64_t n,
    std::complex<float> const* SA, int64_t ldsa,
    std::complex<double>* A, int64_t lda )
{
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldsa_ = to_lapack_int( ldsa );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    LAPACK_clag2z(
        &m_, &n_,
        (lapack_complex_float*) SA, &ldsa_,
        (lapack_complex_double*) A, &lda_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
