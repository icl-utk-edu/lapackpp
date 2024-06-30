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
/// @ingroup geev_computational
int64_t unghr(
    int64_t n, int64_t ilo, int64_t ihi,
    std::complex<float>* A, int64_t lda,
    std::complex<float> const* tau )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int ilo_ = to_lapack_int( ilo );
    lapack_int ihi_ = to_lapack_int( ihi );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_cunghr(
        &n_, &ilo_, &ihi_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );

    LAPACK_cunghr(
        &n_, &ilo_, &ihi_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Generates an n-by-n unitary matrix Q which is defined as the
/// product of ihi-ilo elementary reflectors of order n, as returned by
/// `lapack::gehrd`:
/// \[
///     Q = H(ilo) H(ilo+1) \dots H(ihi-1).
/// \]
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this is an alias for `lapack::orghr`.
///
/// @param[in] n
///     The order of the matrix Q. n >= 0.
///
/// @param[in] ilo
///
/// @param[in] ihi
///     ilo and ihi must have the same values as in the previous call
///     of `lapack::gehrd`. Q is equal to the unit matrix except in the
///     submatrix Q(ilo+1:ihi,ilo+1:ihi).
///     - If n > 0, then 1 <= ilo <= ihi <= n;
///     - if n = 0, ilo=1 and ihi=0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     On entry, the vectors which define the elementary reflectors,
///     as returned by `lapack::gehrd`.
///     On exit, the n-by-n unitary matrix Q.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[in] tau
///     The vector tau of length n-1.
///     tau(i) must contain the scalar factor of the elementary
///     reflector H(i), as returned by `lapack::gehrd`.
///
/// @return = 0: successful exit
///
/// @ingroup geev_computational
int64_t unghr(
    int64_t n, int64_t ilo, int64_t ihi,
    std::complex<double>* A, int64_t lda,
    std::complex<double> const* tau )
{
    lapack_int n_ = to_lapack_int( n );
    lapack_int ilo_ = to_lapack_int( ilo );
    lapack_int ihi_ = to_lapack_int( ihi );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_zunghr(
        &n_, &ilo_, &ihi_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );

    LAPACK_zunghr(
        &n_, &ilo_, &ihi_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
