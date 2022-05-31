// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#if LAPACK_VERSION >= 30600  // >= v3.6

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gesvd
int64_t gesvdx(
    lapack::Job jobu, lapack::Job jobvt, lapack::Range range, int64_t m, int64_t n,
    float* A, int64_t lda, float vl, float vu, int64_t il, int64_t iu,
    int64_t* nfound,
    float* S,
    float* U, int64_t ldu,
    float* VT, int64_t ldvt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(il) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(iu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvt) > std::numeric_limits<lapack_int>::max() );
    }
    char jobu_ = job2char( jobu );
    char jobvt_ = job2char( jobvt );
    char range_ = range2char( range );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int il_ = (lapack_int) il;
    lapack_int iu_ = (lapack_int) iu;
    lapack_int nfound_ = (lapack_int) *nfound;
    lapack_int ldu_ = (lapack_int) ldu;
    lapack_int ldvt_ = (lapack_int) ldvt;
    lapack_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_sgesvdx(
        &jobu_, &jobvt_, &range_, &m_, &n_,
        A, &lda_, &vl, &vu, &il_, &iu_, &nfound_,
        S,
        U, &ldu_,
        VT, &ldvt_,
        qry_work, &ineg_one,
        qry_iwork, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< float > work( lwork_ );
    lapack::vector< lapack_int > iwork( (12*min(m,n)) );

    LAPACK_sgesvdx(
        &jobu_, &jobvt_, &range_, &m_, &n_,
        A, &lda_, &vl, &vu, &il_, &iu_, &nfound_,
        S,
        U, &ldu_,
        VT, &ldvt_,
        &work[0], &lwork_,
        &iwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    *nfound = nfound_;
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesvd
int64_t gesvdx(
    lapack::Job jobu, lapack::Job jobvt, lapack::Range range, int64_t m, int64_t n,
    double* A, int64_t lda, double vl, double vu, int64_t il, int64_t iu,
    int64_t* nfound,
    double* S,
    double* U, int64_t ldu,
    double* VT, int64_t ldvt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(il) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(iu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvt) > std::numeric_limits<lapack_int>::max() );
    }
    char jobu_ = job2char( jobu );
    char jobvt_ = job2char( jobvt );
    char range_ = range2char( range );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int il_ = (lapack_int) il;
    lapack_int iu_ = (lapack_int) iu;
    lapack_int nfound_ = (lapack_int) *nfound;
    lapack_int ldu_ = (lapack_int) ldu;
    lapack_int ldvt_ = (lapack_int) ldvt;
    lapack_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_dgesvdx(
        &jobu_, &jobvt_, &range_, &m_, &n_,
        A, &lda_, &vl, &vu, &il_, &iu_, &nfound_,
        S,
        U, &ldu_,
        VT, &ldvt_,
        qry_work, &ineg_one,
        qry_iwork, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< double > work( lwork_ );
    lapack::vector< lapack_int > iwork( (12*min(m,n)) );

    LAPACK_dgesvdx(
        &jobu_, &jobvt_, &range_, &m_, &n_,
        A, &lda_, &vl, &vu, &il_, &iu_, &nfound_,
        S,
        U, &ldu_,
        VT, &ldvt_,
        &work[0], &lwork_,
        &iwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    *nfound = nfound_;
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesvd
int64_t gesvdx(
    lapack::Job jobu, lapack::Job jobvt, lapack::Range range, int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda, float vl, float vu, int64_t il, int64_t iu,
    int64_t* nfound,
    float* S,
    std::complex<float>* U, int64_t ldu,
    std::complex<float>* VT, int64_t ldvt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(il) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(iu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvt) > std::numeric_limits<lapack_int>::max() );
    }
    char jobu_ = job2char( jobu );
    char jobvt_ = job2char( jobvt );
    char range_ = range2char( range );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int il_ = (lapack_int) il;
    lapack_int iu_ = (lapack_int) iu;
    lapack_int nfound_ = (lapack_int) *nfound;
    lapack_int ldu_ = (lapack_int) ldu;
    lapack_int ldvt_ = (lapack_int) ldvt;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_cgesvdx(
        &jobu_, &jobvt_, &range_, &m_, &n_,
        (lapack_complex_float*) A, &lda_, &vl, &vu, &il_, &iu_, &nfound_,
        S,
        (lapack_complex_float*) U, &ldu_,
        (lapack_complex_float*) VT, &ldvt_,
        (lapack_complex_float*) qry_work, &ineg_one,
        qry_rwork,
        qry_iwork, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // from docs
    int64_t lrwork = min(m,n)*(min(m,n)*2 + 15*min(m,n));

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );
    lapack::vector< float > rwork( (max( 1, lrwork )) );
    lapack::vector< lapack_int > iwork( (12*min(m,n)) );

    LAPACK_cgesvdx(
        &jobu_, &jobvt_, &range_, &m_, &n_,
        (lapack_complex_float*) A, &lda_, &vl, &vu, &il_, &iu_, &nfound_,
        S,
        (lapack_complex_float*) U, &ldu_,
        (lapack_complex_float*) VT, &ldvt_,
        (lapack_complex_float*) &work[0], &lwork_,
        &rwork[0],
        &iwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    *nfound = nfound_;
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the singular value decomposition (SVD) of a
/// m-by-n matrix A, optionally computing the left and/or right singular
/// vectors. The SVD is written
/// \[
///     A = U \Sigma V^H,
/// \]
/// where $\Sigma$ is an m-by-n matrix which is zero except for its
/// min(m,n) diagonal elements, U is an m-by-m unitary matrix, and
/// V is an n-by-n unitary matrix. The diagonal elements of SIGMA
/// are the singular values of A; they are real and non-negative, and
/// are returned in descending order. The first min(m,n) columns of
/// U and V are the left and right singular vectors of A.
///
/// `gesvdx` uses an eigenvalue problem for obtaining the SVD, which
/// allows for the computation of a subset of singular values and
/// vectors. See `lapack::bdsvdx` for details.
///
/// Note that the routine returns VT = $V^H$, not V.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] jobu
///     Specifies options for computing all or part of the matrix U:
///     - lapack::Job::Vec:
///         the first min(m,n) columns of U (the left singular
///         vectors) or as specified by range are returned in the array U;
///     - lapack::Job::NoVec:
///         no columns of U (no left singular vectors) are computed.
///
/// @param[in] jobvt
///      Specifies options for computing all or part of the matrix
///      $V^H$:
///     - lapack::Job::Vec:
///         the first min(m,n) rows of $V^H$ (the right singular
///         vectors) or as specified by range are returned in the array VT;
///     - lapack::Job::NoVec:
///         no rows of $V^H$ (no right singular vectors) are computed.
///
/// @param[in] range
///     - lapack::Range::All:
///         all singular values will be found.
///     - lapack::Range::Value:
///         all singular values in the half-open interval (vl,vu] will be found.
///     - lapack::Range::Index:
///         the il-th through iu-th singular values will be found.
///
/// @param[in] m
///     The number of rows of the input matrix A. m >= 0.
///
/// @param[in] n
///     The number of columns of the input matrix A. n >= 0.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     On entry, the m-by-n matrix A.
///     On exit, the contents of A are destroyed.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @param[in] vl
///     If range=Value, the lower bound of the interval to
///     be searched for singular values. vu > vl.
///     Not referenced if range = All or Index.
///
/// @param[in] vu
///     If range=Value, the upper bound of the interval to
///     be searched for singular values. vu > vl.
///     Not referenced if range = All or Index.
///
/// @param[in] il
///     If range=Index, the index of the
///     smallest singular value to be returned.
///     1 <= il <= iu <= min(m,n), if min(m,n) > 0.
///     Not referenced if range = All or Value.
///
/// @param[in] iu
///     If range=Index, the index of the
///     largest singular value to be returned.
///     1 <= il <= iu <= min(m,n), if min(m,n) > 0.
///     Not referenced if range = All or Value.
///
/// @param[out] nfound
///     The total number of singular values found,
///     0 <= nfound <= min(m,n).
///     - If range = All, nfound = min(m,n);
///     - if range = Index, nfound = iu-il+1.
///
/// @param[out] S
///     The vector S of length min(m,n).
///     The singular values of A, sorted so that S(i) >= S(i+1).
///
/// @param[out] U
///     The m-by-nfound matrix U, stored in an ldu-by-ucol array.
///     - If jobu = Vec, U contains columns of U (the left singular
///       vectors, stored columnwise) as specified by range;
///     - if jobu = NoVec, U is not referenced.
///     \n
///     Note: The user must ensure that ucol >= nfound; if range = Value,
///     the exact value of nfound is not known in advance and an upper
///     bound must be used.
///
/// @param[in] ldu
///     The leading dimension of the array U. ldu >= 1;
///     if jobu = Vec, ldu >= m.
///
/// @param[out] VT
///     The nfound-by-n matrix VT, stored in an ldvt-by-n array.
///     - If jobvt = Vec, VT contains the rows of $V^H$
///       (the right singular vectors, stored rowwise) as specified by range;
///     - if jobvt = NoVec, VT is not referenced.
///     \n
///     Note: The user must ensure that ldvt >= nfound; if range = Value,
///     the exact value of nfound is not known in advance and an upper
///     bound must be used.
///
/// @param[in] ldvt
///     The leading dimension of the array VT. ldvt >= 1;
///     if jobvt = Vec, ldvt >= nfound.
///
/// @return = 0: successful exit
/// @return > 0 and <= n: if return value = i, then i eigenvectors failed to
///              converge in `lapack::bdsvdx`/`lapack::stevx`.
/// @return > n: if return value = 2*n + 1, an internal error occurred in
///              `lapack::bdsvdx`
///
/// @ingroup gesvd
int64_t gesvdx(
    lapack::Job jobu, lapack::Job jobvt, lapack::Range range, int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda, double vl, double vu, int64_t il, int64_t iu,
    int64_t* nfound,
    double* S,
    std::complex<double>* U, int64_t ldu,
    std::complex<double>* VT, int64_t ldvt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(il) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(iu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvt) > std::numeric_limits<lapack_int>::max() );
    }
    char jobu_ = job2char( jobu );
    char jobvt_ = job2char( jobvt );
    char range_ = range2char( range );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int il_ = (lapack_int) il;
    lapack_int iu_ = (lapack_int) iu;
    lapack_int nfound_ = (lapack_int) *nfound;
    lapack_int ldu_ = (lapack_int) ldu;
    lapack_int ldvt_ = (lapack_int) ldvt;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_zgesvdx(
        &jobu_, &jobvt_, &range_, &m_, &n_,
        (lapack_complex_double*) A, &lda_, &vl, &vu, &il_, &iu_, &nfound_,
        S,
        (lapack_complex_double*) U, &ldu_,
        (lapack_complex_double*) VT, &ldvt_,
        (lapack_complex_double*) qry_work, &ineg_one,
        qry_rwork,
        qry_iwork, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // from docs
    int64_t lrwork = min(m,n)*(min(m,n)*2 + 15*min(m,n));

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );
    lapack::vector< double > rwork( (max( 1, lrwork )) );
    lapack::vector< lapack_int > iwork( (12*min(m,n)) );

    LAPACK_zgesvdx(
        &jobu_, &jobvt_, &range_, &m_, &n_,
        (lapack_complex_double*) A, &lda_, &vl, &vu, &il_, &iu_, &nfound_,
        S,
        (lapack_complex_double*) U, &ldu_,
        (lapack_complex_double*) VT, &ldvt_,
        (lapack_complex_double*) &work[0], &lwork_,
        &rwork[0],
        &iwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    *nfound = nfound_;
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= v3.6
