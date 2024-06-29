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
/// @ingroup heev
int64_t heevr(
    lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda, float vl, float vu, int64_t il, int64_t iu, float abstol,
    int64_t* nfound,
    float* W,
    std::complex<float>* Z, int64_t ldz,
    int64_t* isuppz )
{
    char jobz_ = to_char( jobz );
    char range_ = to_char( range );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int il_ = to_lapack_int( il );
    lapack_int iu_ = to_lapack_int( iu );
    lapack_int nfound_ = to_lapack_int( *nfound );
    lapack_int ldz_ = to_lapack_int( ldz );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > isuppz_( (2*max( 1, n )) );  // was max(1,nfound), n >= nfound
        lapack_int* isuppz_ptr = &isuppz_[0];
    #else
        lapack_int* isuppz_ptr = isuppz;
    #endif
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_cheevr(
        &jobz_, &range_, &uplo_, &n_,
        (lapack_complex_float*) A, &lda_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        (lapack_complex_float*) Z, &ldz_,
        isuppz_ptr,
        (lapack_complex_float*) qry_work, &ineg_one,
        qry_rwork, &ineg_one,
        qry_iwork, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);
    lapack_int lrwork_ = real(qry_rwork[0]);
    lapack_int liwork_ = real(qry_iwork[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );
    lapack::vector< float > rwork( lrwork_ );
    lapack::vector< lapack_int > iwork( liwork_ );

    LAPACK_cheevr(
        &jobz_, &range_, &uplo_, &n_,
        (lapack_complex_float*) A, &lda_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        (lapack_complex_float*) Z, &ldz_,
        isuppz_ptr,
        (lapack_complex_float*) &work[0], &lwork_,
        &rwork[0], &lrwork_,
        &iwork[0], &liwork_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    *nfound = nfound_;
    #ifndef LAPACK_ILP64
        std::copy( isuppz_.begin(), isuppz_.end(), isuppz );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes selected eigenvalues and, optionally, eigenvectors
/// of a Hermitian matrix A. Eigenvalues and eigenvectors can
/// be selected by specifying either a range of values or a range of
/// indices for the desired eigenvalues.
///
/// `heevr` first reduces the matrix A to tridiagonal form T with a call
/// to `lapack::hetrd`. Then, whenever possible, `heevr` calls `lapack::stemr` to compute
/// eigenspectrum using Relatively Robust Representations. `lapack::stemr`
/// computes eigenvalues by the dqds algorithm, while orthogonal
/// eigenvectors are computed from various "good" $L D L^T$ representations
/// (also known as Relatively Robust Representations). Gram-Schmidt
/// orthogonalization is avoided as far as possible. More specifically,
/// the various steps of the algorithm are as follows.
///
/// For each unreduced block (submatrix) of T,
///
/// (a) Compute $T - \sigma I = L D L^T$, so that L and D
/// define all the wanted eigenvalues to high relative accuracy.
/// This means that small relative changes in the entries of D and L
/// cause only small relative changes in the eigenvalues and
/// eigenvectors. The standard (unfactored) representation of the
/// tridiagonal matrix T does not have this property in general.
///
/// (b) Compute the eigenvalues to suitable accuracy.
/// If the eigenvectors are desired, the algorithm attains full
/// accuracy of the computed eigenvalues only right before
/// the corresponding vectors have to be computed, see steps c) and d).
///
/// (c) For each cluster of close eigenvalues, select a new
/// shift close to the cluster, find a new factorization, and refine
/// the shifted eigenvalues to suitable accuracy.
///
/// (d) For each eigenvalue with a large enough relative separation compute
/// the corresponding eigenvector by forming a rank revealing twisted
/// factorization. Go back to (c) for any clusters that remain.
///
/// The desired accuracy of the output can be specified by the input
/// parameter abstol.
///
/// For more details, see `lapack::stemr` documentation and:
/// - Inderjit S. Dhillon and Beresford n. Parlett: "Multiple representations
///   to compute orthogonal eigenvectors of symmetric tridiagonal matrices,"
///   Linear Algebra and its Applications, 387(1), pp. 1-28, August 2004.
/// - Inderjit Dhillon and Beresford Parlett: "Orthogonal Eigenvectors and
///   Relative Gaps," SIAM Journal on Matrix Analysis and Applications, Vol. 25,
///   2004. Also LAPACK Working Note 154.
/// - Inderjit Dhillon: "A new O(n^2) algorithm for the symmetric
///   tridiagonal eigenvalue/eigenvector problem",
///   Computer Science Division Technical Report No. UCB/CSD-97-971,
///   UC Berkeley, May 1997.
///
/// Note 1 : `heevr` calls `lapack::stemr` when the full spectrum is requested
/// on machines which conform to the ieee-754 floating point standard.
/// `heevr` calls `lapack::stebz` and `lapack::stein` on non-IEEE machines and
/// when partial spectrum requests are made.
///
/// Normal execution of `lapack::stemr` may create NaNs and infinities and
/// hence may abort due to a floating point exception in environments
/// which do not handle NaNs and infinities in the IEEE standard default
/// manner.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this is an alias for `lapack::syevr`.
///
/// @param[in] jobz
///     - lapack::Job::NoVec: Compute eigenvalues only;
///     - lapack::Job::Vec:   Compute eigenvalues and eigenvectors.
///
/// @param[in] range
///     - lapack::Range::All:
///             all eigenvalues will be found.
///     - lapack::Range::Value:
///             all eigenvalues in the half-open interval (vl,vu] will be found.
///     - lapack::Range::Index:
///             the il-th through iu-th eigenvalues will be found.
///     For range = Value or Index and iu - il < n - 1, `lapack::stebz` and
///     `lapack::stein` are called.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangle of A is stored;
///     - lapack::Uplo::Lower: Lower triangle of A is stored.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     On entry, the Hermitian matrix A.
///     - If uplo = Upper, the
///     leading n-by-n upper triangular part of A contains the
///     upper triangular part of the matrix A.
///
///     - If uplo = Lower,
///     the leading n-by-n lower triangular part of A contains
///     the lower triangular part of the matrix A.
///
///     - On exit, the lower triangle (if uplo=Lower) or the upper
///     triangle (if uplo=Upper) of A, including the diagonal, is
///     destroyed.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[in] vl
///     If range=Value, the lower bound of the interval to
///     be searched for eigenvalues. vl < vu.
///     Not referenced if range = All or Index.
///
/// @param[in] vu
///     If range=Value, the upper bound of the interval to
///     be searched for eigenvalues. vl < vu.
///     Not referenced if range = All or Index.
///
/// @param[in] il
///     If range=Index, the index of the
///     smallest eigenvalue to be returned.
///     1 <= il <= iu <= n, if n > 0; il = 1 and iu = 0 if n = 0.
///     Not referenced if range = All or Value.
///
/// @param[in] iu
///     If range=Index, the index of the
///     largest eigenvalue to be returned.
///     1 <= il <= iu <= n, if n > 0; il = 1 and iu = 0 if n = 0.
///     Not referenced if range = All or Value.
///
/// @param[in] abstol
///     The absolute error tolerance for the eigenvalues.
///     An approximate eigenvalue is accepted as converged
///     when it is determined to lie in an interval [a,b]
///     of width less than or equal to
///     \n
///         abstol + eps * max( |a|,|b| ),
///     \n
///     where eps is the machine precision. If abstol is less than
///     or equal to zero, then eps*|T| will be used in its place,
///     where |T| is the 1-norm of the tridiagonal matrix obtained
///     by reducing A to tridiagonal form.
///     \n
///     See "Computing Small Singular Values of Bidiagonal Matrices
///     with Guaranteed High Relative Accuracy," by Demmel and
///     Kahan, LAPACK Working Note #3.
///     \n
///     If high relative accuracy is important, set abstol to
///     DLAMCH( 'Safe minimum' ). Doing so will guarantee that
///     eigenvalues are computed to high relative accuracy when
///     possible in future releases. The current code does not
///     make any guarantees about high relative accuracy, but
///     future releases will. See J. Barlow and J. Demmel,
///     "Computing Accurate Eigensystems of Scaled Diagonally
///     Dominant Matrices", LAPACK Working Note #7, for a discussion
///     of which matrices define their eigenvalues to high relative
///     accuracy.
///
/// @param[out] nfound
///     The total number of eigenvalues found. 0 <= nfound <= n.
///     - If range = All, nfound = n;
///     - if range = Index, nfound = iu-il+1.
///
/// @param[out] W
///     The vector W of length n.
///     The first nfound elements contain the selected eigenvalues in
///     ascending order.
///
/// @param[out] Z
///     The n-by-nfound matrix Z, stored in an ldz-by-zcol array.
///     - If jobz = Vec, then if successful, the first nfound columns of Z
///     contain the orthonormal eigenvectors of the matrix A
///     corresponding to the selected eigenvalues, with the i-th
///     column of Z holding the eigenvector associated with W(i).
///     - If jobz = NoVec, then Z is not referenced.
///     \n
///     Note: the user must ensure that zcol >= max(1,nfound) columns are
///     supplied in the array Z; if range = Value, the exact value of nfound
///     is not known in advance and an upper bound must be used.
///
/// @param[in] ldz
///     The leading dimension of the array Z. ldz >= 1, and if
///     jobz = Vec, ldz >= max(1,n).
///
/// @param[out] isuppz
///     The vector isuppz of length 2*max(1,nfound).
///     The support of the eigenvectors in Z, i.e., the indices
///     indicating the nonzero elements in Z. The i-th eigenvector
///     is nonzero only in elements isuppz( 2*i-1 ) through
///     isuppz( 2*i ). This is an output of `lapack::stemr` (tridiagonal
///     matrix). The support of the eigenvectors of A is typically
///     1:n because of the unitary transformations applied by `lapack::unmtr`.
///     Implemented only for range = All or Index and iu - il = n - 1
///
/// @return = 0: successful exit
/// @return > 0: Internal error
///
/// @ingroup heev
int64_t heevr(
    lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda, double vl, double vu, int64_t il, int64_t iu, double abstol,
    int64_t* nfound,
    double* W,
    std::complex<double>* Z, int64_t ldz,
    int64_t* isuppz )
{
    char jobz_ = to_char( jobz );
    char range_ = to_char( range );
    char uplo_ = to_char( uplo );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int il_ = to_lapack_int( il );
    lapack_int iu_ = to_lapack_int( iu );
    lapack_int nfound_ = to_lapack_int( *nfound );
    lapack_int ldz_ = to_lapack_int( ldz );
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > isuppz_( (2*max( 1, n )) );  // was max(1,nfound), n >= nfound
        lapack_int* isuppz_ptr = &isuppz_[0];
    #else
        lapack_int* isuppz_ptr = isuppz;
    #endif
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_zheevr(
        &jobz_, &range_, &uplo_, &n_,
        (lapack_complex_double*) A, &lda_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        (lapack_complex_double*) Z, &ldz_,
        isuppz_ptr,
        (lapack_complex_double*) qry_work, &ineg_one,
        qry_rwork, &ineg_one,
        qry_iwork, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);
    lapack_int lrwork_ = real(qry_rwork[0]);
    lapack_int liwork_ = real(qry_iwork[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );
    lapack::vector< double > rwork( lrwork_ );
    lapack::vector< lapack_int > iwork( liwork_ );

    LAPACK_zheevr(
        &jobz_, &range_, &uplo_, &n_,
        (lapack_complex_double*) A, &lda_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        (lapack_complex_double*) Z, &ldz_,
        isuppz_ptr,
        (lapack_complex_double*) &work[0], &lwork_,
        &rwork[0], &lrwork_,
        &iwork[0], &liwork_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    *nfound = nfound_;
    #ifndef LAPACK_ILP64
        std::copy( isuppz_.begin(), isuppz_.end(), isuppz );
    #endif
    return info_;
}

}  // namespace lapack
