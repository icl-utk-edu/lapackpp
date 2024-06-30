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
int64_t trexc(
    lapack::Job compq, int64_t n,
    float* T, int64_t ldt,
    float* Q, int64_t ldq,
    int64_t* ifst,
    int64_t* ilst )
{
    char compq_ = to_char_comp( compq );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int ifst_ = to_lapack_int( *ifst );  // in,out
    lapack_int ilst_ = to_lapack_int( *ilst );  // in,out
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (n) );

    LAPACK_strexc(
        &compq_, &n_,
        T, &ldt_,
        Q, &ldq_, &ifst_, &ilst_,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    *ifst = ifst_;
    *ilst = ilst_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t trexc(
    lapack::Job compq, int64_t n,
    double* T, int64_t ldt,
    double* Q, int64_t ldq,
    int64_t* ifst,
    int64_t* ilst )
{
    char compq_ = to_char_comp( compq );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int ifst_ = to_lapack_int( *ifst );  // in,out
    lapack_int ilst_ = to_lapack_int( *ilst );  // in,out
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (n) );

    LAPACK_dtrexc(
        &compq_, &n_,
        T, &ldt_,
        Q, &ldq_, &ifst_, &ilst_,
        &work[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    *ifst = ifst_;
    *ilst = ilst_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t trexc(
    lapack::Job compq, int64_t n,
    std::complex<float>* T, int64_t ldt,
    std::complex<float>* Q, int64_t ldq, int64_t ifst, int64_t ilst )
{
    char compq_ = to_char_comp( compq );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int ldq_ = to_lapack_int( ldq );
    // In complex, ifst, ilst are in only, not in,out.
    lapack_int ifst_ = to_lapack_int( ifst );
    lapack_int ilst_ = to_lapack_int( ilst );
    lapack_int info_ = 0;

    LAPACK_ctrexc(
        &compq_, &n_,
        (lapack_complex_float*) T, &ldt_,
        (lapack_complex_float*) Q, &ldq_, &ifst_, &ilst_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t trexc(
    lapack::Job compq, int64_t n,
    std::complex<double>* T, int64_t ldt,
    std::complex<double>* Q, int64_t ldq, int64_t ifst, int64_t ilst )
{
    char compq_ = to_char_comp( compq );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ldt_ = to_lapack_int( ldt );
    lapack_int ldq_ = to_lapack_int( ldq );
    // In complex, ifst, ilst are in only, not in,out.
    lapack_int ifst_ = to_lapack_int( ifst );
    lapack_int ilst_ = to_lapack_int( ilst );
    lapack_int info_ = 0;

    LAPACK_ztrexc(
        &compq_, &n_,
        (lapack_complex_double*) T, &ldt_,
        (lapack_complex_double*) Q, &ldq_, &ifst_, &ilst_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
