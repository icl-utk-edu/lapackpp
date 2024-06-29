// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::real;
using blas::is_complex;
using blas::real_type;
using blas::complex_type;

//==============================================================================
namespace internal {

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, float version.
/// @ingroup gges_internal
inline void tgsen(
    lapack_int ijob, lapack_int wantq, lapack_int wantz,
    lapack_logical const* select, lapack_int n,
    float* A, lapack_int lda,
    float* B, lapack_int ldb,
    float* alphar, float* alphai,
    float* beta,
    float* Q, lapack_int ldq,
    float* Z, lapack_int ldz,
    lapack_int* sdim, float* pl, float* pr, float* dif,
    float* work, lapack_int lwork,
    lapack_int* iwork, lapack_int liwork,
    lapack_int* info )
{
    LAPACK_stgsen(
        &ijob, &wantq, &wantz, select, &n,
        A, &lda, B, &ldb, alphar, alphai, beta,
        Q, &ldq, Z, &ldz, sdim, pl, pr, dif,
        work, &lwork, iwork, &liwork, info );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, double version.
/// @ingroup gges_internal
inline void tgsen(
    lapack_int ijob, lapack_int wantq, lapack_int wantz,
    lapack_logical const* select, lapack_int n,
    double* A, lapack_int lda,
    double* B, lapack_int ldb,
    double* alphar, double* alphai,
    double* beta,
    double* Q, lapack_int ldq,
    double* Z, lapack_int ldz,
    lapack_int* sdim, double* pl, double* pr, double* dif,
    double* work, lapack_int lwork,
    lapack_int* iwork, lapack_int liwork,
    lapack_int* info )
{
    LAPACK_dtgsen(
        &ijob, &wantq, &wantz, select, &n,
        A, &lda, B, &ldb, alphar, alphai, beta,
        Q, &ldq, Z, &ldz, sdim, pl, pr, dif,
        work, &lwork, iwork, &liwork, info );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<float> version.
/// @ingroup gges_internal
inline void tgsen(
    lapack_int ijob, lapack_int wantq, lapack_int wantz,
    lapack_logical const* select, lapack_int n,
    std::complex<float>* A, lapack_int lda,
    std::complex<float>* B, lapack_int ldb,
    std::complex<float>* alpha,
    std::complex<float>* beta,
    std::complex<float>* Q, lapack_int ldq,
    std::complex<float>* Z, lapack_int ldz,
    lapack_int* sdim, float* pl, float* pr, float* dif,
    std::complex<float>* work, lapack_int lwork,
    lapack_int* iwork, lapack_int liwork,
    lapack_int* info )
{
    LAPACK_ctgsen(
        &ijob, &wantq, &wantz,
        select, &n,
        (lapack_complex_float*) A, &lda,
        (lapack_complex_float*) B, &ldb,
        (lapack_complex_float*) alpha,
        (lapack_complex_float*) beta,
        (lapack_complex_float*) Q, &ldq,
        (lapack_complex_float*) Z, &ldz, sdim, pl, pr, dif,
        (lapack_complex_float*) work, &lwork,
        iwork, &liwork, info );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double> version.
/// @ingroup gges_internal
inline void tgsen(
    lapack_int ijob, lapack_int wantq, lapack_int wantz,
    lapack_logical const* select, lapack_int n,
    std::complex<double>* A, lapack_int lda,
    std::complex<double>* B, lapack_int ldb,
    std::complex<double>* alpha,
    std::complex<double>* beta,
    std::complex<double>* Q, lapack_int ldq,
    std::complex<double>* Z, lapack_int ldz,
    lapack_int* sdim,
    double* pl, double* pr,
    double* dif,
    std::complex<double>* work, lapack_int lwork,
    lapack_int* iwork, lapack_int liwork,
    lapack_int* info )
{
    LAPACK_ztgsen(
        &ijob, &wantq, &wantz, select, &n,
        (lapack_complex_double*) A, &lda,
        (lapack_complex_double*) B, &ldb,
        (lapack_complex_double*) alpha,
        (lapack_complex_double*) beta,
        (lapack_complex_double*) Q, &ldq,
        (lapack_complex_double*) Z, &ldz, sdim, pl, pr, dif,
        (lapack_complex_double*) work, &lwork,
        iwork, &liwork, info );
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
int64_t tgsen(
    int64_t ijob, bool wantq, bool wantz,
    lapack_logical const* select, int64_t n,
    scalar_t* A, int64_t lda,
    scalar_t* B, int64_t ldb,
    complex_type<scalar_t>* alpha,
    scalar_t* beta,
    scalar_t* Q, int64_t ldq,
    scalar_t* Z, int64_t ldz,
    int64_t* sdim,
    real_type<scalar_t>* pl, real_type<scalar_t>* pr,
    real_type<scalar_t>* dif )
{
    // convert arguments
    if (sizeof(int64_t) > sizeof(lapack_int)) {
    }
    lapack_int ijob_ = to_lapack_int( ijob );
    lapack_int wantq_ = to_lapack_int( wantq );
    lapack_int wantz_ = to_lapack_int( wantz );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldb_ = to_lapack_int( ldb );
    lapack_int ldq_ = to_lapack_int( ldq );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int sdim_ = to_lapack_int( *sdim );
    lapack_int info_ = 0;

    // For real, create vectors for split-complex representation.
    // For complex, creates as dummy `int` type to be optimized away.
    std::conditional_t< is_complex<scalar_t>::value, int, std::vector<scalar_t> >
        alphar, alphai;
    blas_unused( alphar );  // unused in complex
    blas_unused( alphai );

    // query for workspace size
    scalar_t qry_work[ 1 ];
    lapack_int qry_iwork[ 1 ];
    lapack_int ineg_one = -1;
    if constexpr (! is_complex<scalar_t>::value) {
        // For real, use split-complex alpha.
        alphar.resize( n );
        alphai.resize( n );
        internal::tgsen(
            ijob_, wantq_, wantz_, select, n_,
            A, lda_, B, ldb_, &alphar[ 0 ], &alphai[ 0 ], beta,
            Q, ldq_, Z, ldz_, &sdim_, pl, pr, dif,
            qry_work, ineg_one, qry_iwork, ineg_one, &info_ );
    }
    else {
        internal::tgsen(
            ijob_, wantq_, wantz_, select, n_,
            A, lda_, B, ldb_, alpha, beta,
            Q, ldq_, Z, ldz_, &sdim_, pl, pr, dif,
            qry_work, ineg_one, qry_iwork, ineg_one, &info_ );
    }
    if (info_ < 0) {
        throw Error();
    }
    // LAPACK <= 3.11 has query & documentation error in workspace size; add 1.
    lapack_int lwork_  = real( qry_work[ 0 ] ) + 1;
    lapack_int liwork_ = real( qry_iwork[ 0 ] );

    // allocate workspace
    std::vector< scalar_t > work( lwork_ );
    std::vector< lapack_int > iwork( liwork_ );

    // call low-level wrapper
    if constexpr (! is_complex<scalar_t>::value) {
        // For real, use split-complex alpha.
        internal::tgsen(
            ijob_, wantq_, wantz_, select, n_,
            A, lda_, B, ldb_, &alphar[0], &alphai[0], beta,
            Q, ldq_, Z, ldz_, &sdim_, pl, pr, dif,
            &work[0], lwork_, &iwork[0], liwork_, &info_ );
        // Merge split-complex representation.
        for (int64_t i = 0; i < n; ++i) {
            alpha[ i ] = std::complex<scalar_t>( alphar[ i ], alphai[ i ] );
        }
    }
    else {
        internal::tgsen(
            ijob_, wantq_, wantz_, select, n_,
            A, lda_, B, ldb_, alpha, beta,
            Q, ldq_, Z, ldz_, &sdim_, pl, pr, dif,
            &work[0], lwork_, &iwork[0], liwork_, &info_ );
    }
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

}  // namespace impl

//==============================================================================
/// Reorders the generalized Schur decomposition of a complex
/// matrix pair (A, B) (in terms of an unitary equivalence trans-
/// formation Q^H * (A, B) * Z), so that a selected cluster of eigenvalues
/// appears in the leading diagonal blocks of the pair (A,B). The leading
/// columns of Q and Z form unitary bases of the corresponding left and
/// right eigenspaces (deflating subspaces). (A, B) must be in
/// generalized Schur canonical form, that is, A and B are both upper
/// triangular.
///
/// `tgsen` also computes the generalized eigenvalues
///
///     w(j) = alpha(j) / beta(j)
///
/// of the reordered matrix pair (A, B).
///
/// Optionally, the routine computes estimates of reciprocal condition
/// numbers for eigenvalues and eigenspaces. These are
///     Difu[ (A11, B11), (A22, B22) ]
/// and
///     Difl[ (A11, B11), (A22, B22) ],
/// i.e. the separation(s)
/// between the matrix pairs (A11, B11) and (A22, B22) that correspond to
/// the selected cluster and the eigenvalues outside the cluster, resp.,
/// and norms of "projections" onto left and right eigenspaces w.r.t.
/// the selected cluster in the (1, 1)-block.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] ijob
///     Whether condition numbers are required for the
///     cluster of eigenvalues (pl and pr) or the deflating subspaces
///     (Difu and Difl):
///     * 0: Only reorder w.r.t. select. No extras.
///     * 1: Reciprocal of norms of "projections" onto left and right
///          eigenspaces w.r.t. the selected cluster (pl and pr).
///     * 2: Upper bounds on Difu and Difl. F-norm-based estimate
///          (dif(1:2)).
///     * 3: Estimate of Difu and Difl. 1-norm-based estimate
///          (dif(1:2)).
///          About 5 times as expensive as ijob = 2.
///     * 4: Compute pl, pr and dif (i.e. 0, 1 and 2 above):
///          Economic version to get it all.
///     * 5: Compute pl, pr and dif (i.e. 0, 1 and 3 above)
///
/// @param[in] wantq
///     * true:  update the left transformation matrix Q;
///     * false: do not update Q.
///
/// @param[in] wantz
///     * true:  update the right transformation matrix Z;
///     * false: do not update Z.
///
/// @param[in] select
///     The vector select of length n.
///     select specifies the eigenvalues in the selected cluster. To
///     select an eigenvalue w(j), select(j) must be set to
///     true.
///
/// @param[in] n
///     The order of the matrices A and B. n >= 0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     On entry, the upper triangular matrix A, in generalized
///     Schur canonical form.
///     On exit, A is overwritten by the reordered matrix A.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[in,out] B
///     The n-by-n matrix B, stored in an ldb-by-n array.
///     On entry, the upper triangular matrix B, in generalized
///     Schur canonical form.
///     On exit, B is overwritten by the reordered matrix B.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,n).
///
/// @param[out] alpha
///     The vector alpha of length n.
///
/// @param[out] beta
///     The vector beta of length n.
///     \n
///     The diagonal elements of A and B, respectively,
///     when the pair (A, B) has been reduced to generalized Schur
///     form. alpha(i) / beta(i) for i = 1, ..., n are the generalized
///     eigenvalues.
///
/// @param[in,out] Q
///     The n-by-n matrix Q, stored in an ldq-by-n array.
///     On entry, if wantq = true, Q is an n-by-n matrix.
///     On exit, Q has been postmultiplied by the left unitary
///     transformation matrix which reorder (A, B); The leading sdim
///     columns of Q form orthonormal bases for the specified pair of
///     left eigenspaces (deflating subspaces).
///     If wantq = false, Q is not referenced.
///
/// @param[in] ldq
///     The leading dimension of the array Q. ldq >= 1.
///     If wantq = true, ldq >= n.
///
/// @param[in,out] Z
///     The n-by-n matrix Z, stored in an ldz-by-n array.
///     On entry, if wantz = true, Z is an n-by-n matrix.
///     On exit, Z has been postmultiplied by the left unitary
///     transformation matrix which reorder (A, B); The leading sdim
///     columns of Z form orthonormal bases for the specified pair of
///     left eigenspaces (deflating subspaces).
///     If wantz = false, Z is not referenced.
///
/// @param[in] ldz
///     The leading dimension of the array Z. ldz >= 1.
///     If wantz = true, ldz >= n.
///
/// @param[out] sdim
///     The dimension of the specified pair of left and right
///     eigenspaces (deflating subspaces) 0 <= sdim <= n.
///     (Called `m` in LAPACK.)
///
/// @param[out] pl
/// @param[out] pr
///     * If ijob = 1, 4 or 5, then pl, pr are lower bounds on the
///       reciprocal of the norm of "projections" onto left and right
///       eigenspace with respect to the selected cluster.
///       0 < pl, pr <= 1.
///     * If sdim = 0 or sdim = n, then pl = pr = 1.
///     * If ijob = 0, 2 or 3, then pl, pr are not referenced.
///
/// @param[out] dif
///     The vector dif of length 2.
///     * If ijob >= 2, dif(1:2) store the estimates of Difu and Difl.
///     * If ijob = 2 or 4, dif(1:2) are F-norm-based upper bounds on
///       Difu and Difl.
///     * If ijob = 3 or 5, dif(1:2) are 1-norm-based
///       estimates of Difu and Difl, computed using reversed
///       communication with `lapack::lacn2`.
///     * If sdim = 0 or n, dif(1:2) = F-norm([A, B]).
///     * If ijob = 0 or 1, dif is not referenced.
///
/// @return
/// * 0: Successful exit.
/// * 1: Reordering of (A, B) failed because the transformed
///     matrix pair (A, B) would be too far from generalized
///     Schur form; the problem is very ill-conditioned.
///     (A, B) may have been partially reordered.
///     If requested, 0 is returned in dif(*), pl and pr.
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// `tgsen` first collects the selected eigenvalues by computing unitary
/// U and W that move them to the top left corner of (A, B). In other
/// words, the selected eigenvalues are the eigenvalues of (A11, B11) in
///
///     U^H (A, B) W = (A11 A12)  (B11 B12) n1
///                    ( 0  A22), ( 0  B22) n2
///                      n1  n2     n1  n2
///
/// where n = n1+n2 and U^H means the conjugate transpose of U. The first
/// n1 columns of U and W span the specified pair of left and right
/// eigenspaces (deflating subspaces) of (A, B).
///
/// If (A, B) has been obtained from the generalized real Schur
/// decomposition of a matrix pair (C, D) = Q (A, B) Z^H, then the
/// reordered generalized Schur form of (C, D) is given by
///
///     (C, D) = (QU) (U^H (A, B) W) (ZW)^H,
///
/// and the first n1 columns of QU and ZW span the corresponding
/// deflating subspaces of (C, D) (Q and Z store QU and ZW, resp.).
///
/// Note that if the selected eigenvalue is sufficiently ill-conditioned,
/// then its value may differ significantly from its value before
/// reordering.
///
/// The reciprocal condition numbers of the left and right eigenspaces
/// spanned by the first n1 columns of U and W (or QU and ZW) may
/// be returned in dif(1:2), corresponding to Difu and Difl, resp.
///
/// The Difu and Difl are defined as:
///
///     Difu[(A11, B11), (A22, B22)] = sigma-min( Zu )
/// and
///     Difl[(A11, B11), (A22, B22)] = Difu[(A22, B22), (A11, B11)],
///
/// where sigma-min(Zu) is the smallest singular value of the
/// (2 n1 n2)-by-(2 n1 n2) matrix
///
///     Zu = [ kron( In2, A11 )  -kron( A22^H, In1 ) ]
///          [ kron( In2, B11 )  -kron( B22^H, In1 ) ].
///
/// Here, Inx is the identity matrix of size nx and A22^H is the
/// conjugate transpose of A22. kron(X, Y) is the Kronecker product between
/// the matrices X and Y.
///
/// When dif(2) is small, small changes in (A, B) can cause large changes
/// in the deflating subspace. An approximate (asymptotic) bound on the
/// maximum angular error in the computed deflating subspaces is
///
///     EPS * norm((A, B)) / dif(2),
///
/// where EPS is the machine precision.
///
/// The reciprocal norm of the projectors on the left and right
/// eigenspaces associated with (A11, B11) may be returned in pl and pr.
/// They are computed as follows. First we compute L and R so that
/// P*(A, B)*Q is block diagonal, where
///
///     P = ( I -L ) n1           Q = ( I R ) n1
///         ( 0  I ) n2    and        ( 0 I ) n2
///           n1 n2                    n1 n2
///
/// and (L, R) is the solution to the generalized Sylvester equation
///
///     A11*R - L*A22 = -A12
///     B11*R - L*B22 = -B12
///
/// Then pl = (F-norm(L)^2+1)^(-1/2) and pr = (F-norm(R)^2+1)^(-1/2).
/// An approximate (asymptotic) bound on the average absolute error of
/// the selected eigenvalues is
///
///     EPS * norm((A, B)) / pl.
///
/// There are also global error bounds which valid for perturbations up
/// to a certain restriction:  A lower bound (x) on the smallest
/// F-norm(E,F) for which an eigenvalue of (A11, B11) may move and
/// coalesce with an eigenvalue of (A22, B22) under perturbation (E,F),
/// (i.e. (A + E, B + F), is
///
///  x = min(Difu,Difl)/((1/(pl*pl)+1/(pr*pr))^(1/2)+2*max(1/pl,1/pr)).
///
/// An approximate bound on x can be computed from dif(1:2), pl and pr.
///
/// If y = ( F-norm(E,F) / x) <= 1, the angles between the perturbed
/// (L', R') and unperturbed (L, R) left and right deflating subspaces
/// associated with the selected cluster in the (1,1)-blocks can be
/// bounded as
///
///  max-angle(L, L') <= arctan( y * pl / (1 - y * (1 - pl * pl)^(1/2))
///  max-angle(R, R') <= arctan( y * pr / (1 - y * (1 - pr * pr)^(1/2))
///
/// See LAPACK User's Guide section 4.11 or the following references
/// for more information.
///
/// Note that if the default method for computing the Frobenius-norm-
/// based estimate dif is not wanted (see `lapack::latdf`), then the parameter
/// IDIFJB (see below) should be changed from 3 to 4 (routine `lapack::latdf`
/// (ijob = 2 will be used)). See `lapack::tgsyl` for more details.
///
//------------------------------------------------------------------------------
/// High-level overloaded wrapper, float version.
/// @ingroup gges
int64_t tgsen(
    int64_t ijob, bool wantq, bool wantz,
    lapack_logical const* select, int64_t n,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    std::complex<float>* alpha,
    float* beta,
    float* Q, int64_t ldq,
    float* Z, int64_t ldz,
    int64_t* sdim,
    float* pl, float* pr,
    float* dif )
{
    return impl::tgsen(
        ijob, wantq, wantz, select, n,
        A, lda, B, ldb, alpha, beta,
        Q, ldq, Z, ldz, sdim, pl, pr, dif );
}

//------------------------------------------------------------------------------
/// High-level overloaded wrapper, double version.
/// @ingroup gges
int64_t tgsen(
    int64_t ijob, bool wantq, bool wantz,
    lapack_logical const* select, int64_t n,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    std::complex<double>* alpha,
    double* beta,
    double* Q, int64_t ldq,
    double* Z, int64_t ldz,
    int64_t* sdim,
    double* pl, double* pr,
    double* dif )
{
    return impl::tgsen(
        ijob, wantq, wantz, select, n,
        A, lda, B, ldb, alpha, beta,
        Q, ldq, Z, ldz, sdim, pl, pr, dif );
}

//------------------------------------------------------------------------------
/// High-level overloaded wrapper, complex<float> version.
/// @ingroup gges
int64_t tgsen(
    int64_t ijob, bool wantq, bool wantz,
    lapack_logical const* select, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* alpha,
    std::complex<float>* beta,
    std::complex<float>* Q, int64_t ldq,
    std::complex<float>* Z, int64_t ldz,
    int64_t* sdim,
    float* pl, float* pr,
    float* dif )
{
    return impl::tgsen(
        ijob, wantq, wantz, select, n,
        A, lda, B, ldb, alpha, beta,
        Q, ldq, Z, ldz, sdim, pl, pr, dif );
}

//------------------------------------------------------------------------------
/// High-level overloaded wrapper, complex<double> version.
/// @ingroup gges
int64_t tgsen(
    int64_t ijob, bool wantq, bool wantz,
    lapack_logical const* select, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* alpha,
    std::complex<double>* beta,
    std::complex<double>* Q, int64_t ldq,
    std::complex<double>* Z, int64_t ldz,
    int64_t* sdim,
    double* pl, double* pr,
    double* dif )
{
    return impl::tgsen(
        ijob, wantq, wantz, select, n,
        A, lda, B, ldb, alpha, beta,
        Q, ldq, Z, ldz, sdim, pl, pr, dif );
}

}  // namespace lapack
