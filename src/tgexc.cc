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
using blas::is_complex;

//==============================================================================
namespace internal {

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, float version.
/// @ingroup gges_internal
inline void tgexc(
    lapack_int wantq, lapack_int wantz, lapack_int n,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    float* Q, lapack_int ldq,
    float* Z, lapack_int ldz,
    lapack_int* ifst, lapack_int* ilst,
    float* work, lapack_int lwork, lapack_int* info )
{
    LAPACK_stgexc(
        &wantq, &wantz, &n,
        A, &lda,
        B, &ldb,
        Q, &ldq,
        Z, &ldz, ifst, ilst, &work[0], &lwork, info );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, double version.
/// @ingroup gges_internal
inline void tgexc(
    lapack_int wantq, lapack_int wantz, lapack_int n,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    double* Q, lapack_int ldq,
    double* Z, lapack_int ldz,
    lapack_int* ifst, lapack_int* ilst,
    double* work, lapack_int lwork, lapack_int* info )
{
    LAPACK_dtgexc(
        &wantq, &wantz, &n,
        A, &lda,
        B, &ldb,
        Q, &ldq,
        Z, &ldz, ifst, ilst, &work[0], &lwork, info );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<float> version.
/// @ingroup gges_internal
inline void tgexc(
    lapack_int wantq, lapack_int wantz, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* Q, lapack_int ldq,
    std::complex<float>* Z, lapack_int ldz,
    lapack_int* ifst, lapack_int* ilst, lapack_int* info )
{
    // No workspace for complex.
    LAPACK_ctgexc(
        &wantq, &wantz, &n,
        (lapack_complex_float*) A, &lda,
        (lapack_complex_float*) B, &ldb,
        (lapack_complex_float*) Q, &ldq,
        (lapack_complex_float*) Z, &ldz, ifst, ilst, info );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double> version.
/// @ingroup gges_internal
inline void tgexc(
    lapack_int wantq, lapack_int wantz, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* Q, lapack_int ldq,
    std::complex<double>* Z, lapack_int ldz,
    lapack_int* ifst, lapack_int* ilst, lapack_int* info )
{
    // No workspace for complex.
    LAPACK_ztgexc(
        &wantq, &wantz, &n,
        (lapack_complex_double*) A, &lda,
        (lapack_complex_double*) B, &ldb,
        (lapack_complex_double*) Q, &ldq,
        (lapack_complex_double*) Z, &ldz, ifst, ilst, info );
}

}  // namespace internal

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// @ingroup gges_internal
///
template <typename scalar_t>
int64_t tgexc(
    bool wantq, bool wantz, int64_t n,
    scalar_t* A, int64_t lda,
    scalar_t* B, int64_t ldb,
    scalar_t* Q, int64_t ldq,
    scalar_t* Z, int64_t ldz,
    int64_t* ifst, int64_t* ilst )
{
    lapack_int wantq_ = to_lapack_int( wantq );
    lapack_int wantz_ = to_lapack_int( wantz );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int ifst_ = to_lapack_int( *ifst );  // in,out
    lapack_int ilst_ = to_lapack_int( *ilst );  // in,out
    lapack_int info_ = 0;

    if constexpr (! is_complex<scalar_t>::value) {
        // Real needs workspace.
        // query for workspace size
        scalar_t qry_work[1];
        lapack_int ineg_one = -1;
        internal::tgexc(
            wantq_, wantz_, n_,
            A, lda_, B, ldb_, Q, ldq_, Z, ldz_, &ifst_, &ilst_,
            qry_work, ineg_one, &info_ );
        if (info_ < 0) {
            throw Error();
        }
        lapack_int lwork_ = real( qry_work[ 0 ] );

        // allocate workspace
        std::vector< scalar_t > work( lwork_ );

        // call low-level wrapper
        internal::tgexc(
            wantq_, wantz_, n_,
            A, lda_, B, ldb_, Q, ldq_, Z, ldz_, &ifst_, &ilst_,
            &work[0], lwork_, &info_ );
    }
    else {
        // call low-level wrapper
        internal::tgexc(
            wantq_, wantz_, n_,
            A, lda_, B, ldb_, Q, ldq_, Z, ldz_, &ifst_, &ilst_, &info_ );
    }
    if (info_ < 0) {
        throw Error();
    }
    *ifst = ifst_;
    *ilst = ilst_;
    return info_;
}

}  // namespace impl

//==============================================================================
/// Reorders the generalized Schur decomposition of a complex
/// matrix pair (A,B), using an unitary equivalence transformation
/// (A, B) := Q * (A, B) * Z^H, so that the diagonal block of (A, B) with
/// row index ifst is moved to row ilst.
///
/// (A, B) must be in generalized Schur canonical form, that is, A and
/// B are both upper triangular.
///
/// Optionally, the matrices Q and Z of generalized Schur vectors are
/// updated.
///
///     Q(in) * A(in) * Z(in)^H = Q(out) * A(out) * Z(out)^H
///     Q(in) * B(in) * Z(in)^H = Q(out) * B(out) * Z(out)^H
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] wantq
///     - true:  update the left transformation matrix Q;
///     - false: do not update Q.
///
/// @param[in] wantz
///     - true:  update the right transformation matrix Z;
///     - false: do not update Z.
///
/// @param[in] n
///     The order of the matrices A and B. n >= 0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     On entry, the upper triangular matrix A in the pair (A, B).
///     On exit, the updated matrix A.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[in,out] B
///     The n-by-n matrix B, stored in an ldb-by-n array.
///     On entry, the upper triangular matrix B in the pair (A, B).
///     On exit, the updated matrix B.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,n).
///
/// @param[in,out] Q
///     The n-by-n matrix Q, stored in an ldq-by-n array.
///     On entry, if wantq = true, the unitary matrix Q.
///     On exit, the updated matrix Q.
///     If wantq = false, Q is not referenced.
///
/// @param[in] ldq
///     The leading dimension of the array Q. ldq >= 1;
///     If wantq = true, ldq >= n.
///
/// @param[in,out] Z
///     The n-by-n matrix Z, stored in an ldz-by-n array.
///     On entry, if wantz = true, the unitary matrix Z.
///     On exit, the updated matrix Z.
///     If wantz = false, Z is not referenced.
///
/// @param[in] ldz
///     The leading dimension of the array Z. ldz >= 1;
///     If wantz = true, ldz >= n.
///
/// @param[in,out] ifst
/// @param[in,out] ilst
///     Specify the reordering of the diagonal blocks of (A, B).
///     The block with row index ifst is moved to row ilst, by a
///     sequence of swapping between adjacent blocks.
///     \n
///     For the real version:
///     On exit, if ifst pointed on entry to the second row of
///     a 2-by-2 block, it is changed to point to the first row;
///     ilst always points to the first row of the block in its
///     final position (which may differ from its input value by
///     +1 or -1). 1 <= ifst, ilst <= N.
///     \n
///     For the complex version:
///     In LAPACK, ifst is only input [in], instead of input and output
///     [in,out], but LAPACK++ uses a pointer to be consistent with the
///     real routine.
///
/// @return
/// * 0: Successful exit.
/// * 1: The transformed matrix pair (A, B) would be too far
///     from generalized Schur form; the problem is ill-
///     conditioned. (A, B) may have been partially reordered,
///     and ilst points to the first row of the current
///     position of the block being moved.
///
//------------------------------------------------------------------------------
/// High-level overloaded wrapper, float version.
/// @ingroup gges
int64_t tgexc(
    bool wantq, bool wantz, int64_t n,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    float* Q, int64_t ldq,
    float* Z, int64_t ldz,
    int64_t* ifst, int64_t* ilst )
{
    return impl::tgexc(
        wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz, ifst, ilst );
}

//------------------------------------------------------------------------------
/// High-level overloaded wrapper, double version.
/// @ingroup gges
int64_t tgexc(
    bool wantq, bool wantz, int64_t n,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    double* Q, int64_t ldq,
    double* Z, int64_t ldz,
    int64_t* ifst,
    int64_t* ilst )
{
    return impl::tgexc(
        wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz, ifst, ilst );
}

//------------------------------------------------------------------------------
/// High-level overloaded wrapper, complex<float> version.
/// @ingroup gges
int64_t tgexc(
    bool wantq, bool wantz, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* Q, int64_t ldq,
    std::complex<float>* Z, int64_t ldz,
    int64_t* ifst, int64_t* ilst )
{
    return impl::tgexc(
        wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz, ifst, ilst );
}

//------------------------------------------------------------------------------
/// High-level overloaded wrapper, complex<double> version.
/// @ingroup gges
int64_t tgexc(
    bool wantq, bool wantz, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* Q, int64_t ldq,
    std::complex<double>* Z, int64_t ldz,
    int64_t* ifst, int64_t* ilst )
{
    return impl::tgexc(
        wantq, wantz, n, A, lda, B, ldb, Q, ldq, Z, ldz, ifst, ilst );
}

}  // namespace lapack
