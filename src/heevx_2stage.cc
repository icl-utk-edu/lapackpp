// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#if LAPACK_VERSION >= 30700  // >= 3.7

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup heev
int64_t heevx_2stage(
    lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda, float vl, float vu, int64_t il, int64_t iu, float abstol,
    int64_t* nfound,
    float* W,
    std::complex<float>* Z, int64_t ldz,
    int64_t* ifail )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(il) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(iu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char range_ = range2char( range );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int il_ = (lapack_int) il;
    lapack_int iu_ = (lapack_int) iu;
    lapack_int nfound_ = (lapack_int) *nfound;
    lapack_int ldz_ = (lapack_int) ldz;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ifail_( (n) );
        lapack_int* ifail_ptr = &ifail_[0];
    #else
        lapack_int* ifail_ptr = ifail;
    #endif
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_cheevx_2stage(
        &jobz_, &range_, &uplo_, &n_,
        (lapack_complex_float*) A, &lda_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        (lapack_complex_float*) Z, &ldz_,
        (lapack_complex_float*) qry_work, &ineg_one,
        qry_rwork,
        qry_iwork,
        ifail_ptr, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );
    lapack::vector< float > rwork( (7*n) );
    lapack::vector< lapack_int > iwork( (5*n) );

    LAPACK_cheevx_2stage(
        &jobz_, &range_, &uplo_, &n_,
        (lapack_complex_float*) A, &lda_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        (lapack_complex_float*) Z, &ldz_,
        (lapack_complex_float*) &work[0], &lwork_,
        &rwork[0],
        &iwork[0],
        ifail_ptr, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    *nfound = nfound_;
    #ifndef LAPACK_ILP64
        if (jobz != Job::NoVec) {
            std::copy( &ifail_[ 0 ], &ifail_[ nfound_ ], ifail );
        }
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes selected eigenvalues and, optionally, eigenvectors
/// of a Hermitian matrix A using the 2-stage technique for
/// the reduction to tridiagonal. Eigenvalues and eigenvectors can
/// be selected by specifying either a range of values or a range of
/// indices for the desired eigenvalues.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] jobz
///     - lapack::Job::NoVec: Compute eigenvalues only;
///     - lapack::Job::Vec:   Compute eigenvalues and eigenvectors.
///                           Not yet available (as of LAPACK 3.8.0).
///
/// @param[in] range
///     - lapack::Range::All: all eigenvalues will be found.
///     - lapack::Range::Value: all eigenvalues in the half-open interval (vl,vu]
///         will be found.
///     - lapack::Range::Index: the il-th through iu-th eigenvalues will be found.
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
///         abstol + eps * max(|a|, |b|),
///     \n
///     where eps is the machine precision. If abstol is less than
///     or equal to zero, then eps*|T| will be used in its place,
///     where |T| is the 1-norm of the tridiagonal matrix obtained
///     by reducing A to tridiagonal form.
///     \n
///     Eigenvalues will be computed most accurately when abstol is
///     set to twice the underflow threshold 2*DLAMCH('S'), not zero.
///     If this routine returns with return value > 0, indicating that some
///     eigenvectors did not converge, try setting abstol to
///     2*DLAMCH('S').
///     \n
///     See "Computing Small Singular Values of Bidiagonal Matrices
///     with Guaranteed High Relative Accuracy," by Demmel and
///     Kahan, LAPACK Working Note #3.
///
/// @param[out] nfound
///     The total number of eigenvalues found. 0 <= nfound <= n.
///     - If range = All, nfound = n;
///     - if range = Index, nfound = iu-il+1.
///
/// @param[out] W
///     The vector W of length n.
///     On normal exit, the first nfound elements contain the selected
///     eigenvalues in ascending order.
///
/// @param[out] Z
///     The n-by-nfound matrix Z, stored in an ldz-by-zcol array.
///     - If jobz = Vec, then if successful, the first nfound columns of Z
///     contain the orthonormal eigenvectors of the matrix A
///     corresponding to the selected eigenvalues, with the i-th
///     column of Z holding the eigenvector associated with W(i).
///     If an eigenvector fails to converge, then that column of Z
///     contains the latest approximation to the eigenvector, and the
///     index of the eigenvector is returned in ifail.
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
/// @param[out] ifail
///     The vector ifail of length n.
///     - If jobz = Vec, then if successful, the first nfound elements of
///     ifail are zero. If return value > 0, then ifail contains the
///     indices of the eigenvectors that failed to converge.
///     - If jobz = NoVec, then ifail is not referenced.
///
/// @return = 0: successful exit
/// @return > 0: if return value = i, then i eigenvectors failed to converge.
///              Their indices are stored in array ifail.
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// All details about the 2-stage techniques are available in:
///
/// Azzam Haidar, Hatem Ltaief, and Jack Dongarra.
/// Parallel reduction to condensed forms for symmetric eigenvalue problems
/// using aggregated fine-grained and memory-aware kernels. In Proceedings
/// of 2011 International Conference for High Performance Computing,
/// Networking, Storage and Analysis (SC '11), New York, NY, USA,
/// Article 8, 11 pages.
/// http://doi.acm.org/10.1145/2063384.2063394
///
/// A. Haidar, J. Kurzak, P. Luszczek, 2013.
/// An improved parallel singular value algorithm and its implementation
/// for multicore hardware, In Proceedings of 2013 International Conference
/// for High Performance Computing, Networking, Storage and Analysis (SC '13).
/// Denver, Colorado, USA, 2013.
/// Article 90, 12 pages.
/// http://doi.acm.org/10.1145/2503210.2503292
///
/// A. Haidar, R. Solca, S. Tomov, T. Schulthess and J. Dongarra.
/// A novel hybrid CPU-GPU generalized eigensolver for electronic structure
/// calculations based on fine-grained memory aware tasks.
/// International Journal of High Performance Computing Applications.
/// Volume 28 Issue 2, Pages 196-209, May 2014.
/// http://hpc.sagepub.com/content/28/2/196
///
/// @ingroup heev
int64_t heevx_2stage(
    lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda, double vl, double vu, int64_t il, int64_t iu, double abstol,
    int64_t* nfound,
    double* W,
    std::complex<double>* Z, int64_t ldz,
    int64_t* ifail )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(il) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(iu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char range_ = range2char( range );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int il_ = (lapack_int) il;
    lapack_int iu_ = (lapack_int) iu;
    lapack_int nfound_ = (lapack_int) *nfound;
    lapack_int ldz_ = (lapack_int) ldz;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        lapack::vector< lapack_int > ifail_( (n) );
        lapack_int* ifail_ptr = &ifail_[0];
    #else
        lapack_int* ifail_ptr = ifail;
    #endif
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_zheevx_2stage(
        &jobz_, &range_, &uplo_, &n_,
        (lapack_complex_double*) A, &lda_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        (lapack_complex_double*) Z, &ldz_,
        (lapack_complex_double*) qry_work, &ineg_one,
        qry_rwork,
        qry_iwork,
        ifail_ptr, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );
    lapack::vector< double > rwork( (7*n) );
    lapack::vector< lapack_int > iwork( (5*n) );

    LAPACK_zheevx_2stage(
        &jobz_, &range_, &uplo_, &n_,
        (lapack_complex_double*) A, &lda_, &vl, &vu, &il_, &iu_, &abstol, &nfound_,
        W,
        (lapack_complex_double*) Z, &ldz_,
        (lapack_complex_double*) &work[0], &lwork_,
        &rwork[0],
        &iwork[0],
        ifail_ptr, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    *nfound = nfound_;
    #ifndef LAPACK_ILP64
        if (jobz != Job::NoVec) {
            std::copy( &ifail_[ 0 ], &ifail_[ nfound_ ], ifail );
        }
    #endif
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.7
