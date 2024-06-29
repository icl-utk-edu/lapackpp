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
/// @ingroup gesvd
int64_t gesvd(
    lapack::Job jobu, lapack::Job jobvt, int64_t m, int64_t n,
    float* A, int64_t lda,
    float* S,
    float* U, int64_t ldu,
    float* VT, int64_t ldvt )
{
    char jobu_ = to_char( jobu );
    char jobvt_ = to_char( jobvt );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldu_ = to_lapack_int( ldu );
    lapack_int ldvt_ = to_lapack_int( ldvt );
    lapack_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_sgesvd(
        &jobu_, &jobvt_, &m_, &n_,
        A, &lda_,
        S,
        U, &ldu_,
        VT, &ldvt_,
        qry_work, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_sgesvd(
        &jobu_, &jobvt_, &m_, &n_,
        A, &lda_,
        S,
        U, &ldu_,
        VT, &ldvt_,
        &work[0], &lwork_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesvd
int64_t gesvd(
    lapack::Job jobu, lapack::Job jobvt, int64_t m, int64_t n,
    double* A, int64_t lda,
    double* S,
    double* U, int64_t ldu,
    double* VT, int64_t ldvt )
{
    char jobu_ = to_char( jobu );
    char jobvt_ = to_char( jobvt );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldu_ = to_lapack_int( ldu );
    lapack_int ldvt_ = to_lapack_int( ldvt );
    lapack_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_dgesvd(
        &jobu_, &jobvt_, &m_, &n_,
        A, &lda_,
        S,
        U, &ldu_,
        VT, &ldvt_,
        qry_work, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dgesvd(
        &jobu_, &jobvt_, &m_, &n_,
        A, &lda_,
        S,
        U, &ldu_,
        VT, &ldvt_,
        &work[0], &lwork_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesvd
int64_t gesvd(
    lapack::Job jobu, lapack::Job jobvt, int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda,
    float* S,
    std::complex<float>* U, int64_t ldu,
    std::complex<float>* VT, int64_t ldvt )
{
    char jobu_ = to_char( jobu );
    char jobvt_ = to_char( jobvt );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldu_ = to_lapack_int( ldu );
    lapack_int ldvt_ = to_lapack_int( ldvt );
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_cgesvd(
        &jobu_, &jobvt_, &m_, &n_,
        (lapack_complex_float*) A, &lda_,
        S,
        (lapack_complex_float*) U, &ldu_,
        (lapack_complex_float*) VT, &ldvt_,
        (lapack_complex_float*) qry_work, &ineg_one,
        qry_rwork, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );
    std::vector< float > rwork( (5*min(m,n)) );

    LAPACK_cgesvd(
        &jobu_, &jobvt_, &m_, &n_,
        (lapack_complex_float*) A, &lda_,
        S,
        (lapack_complex_float*) U, &ldu_,
        (lapack_complex_float*) VT, &ldvt_,
        (lapack_complex_float*) &work[0], &lwork_,
        &rwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the singular value decomposition (SVD) of a
/// m-by-n matrix A, optionally computing the left and/or right singular
/// vectors. The SVD is written
/// \[
///     A = U \Sigma V^H
/// \]
///
/// where $\Sigma$ is an m-by-n matrix which is zero except for its
/// min(m,n) diagonal elements, U is an m-by-m unitary matrix, and
/// V is an n-by-n unitary matrix. The diagonal elements of $\Sigma$
/// are the singular values of A; they are real and non-negative, and
/// are returned in descending order. The first min(m,n) columns of
/// U and V are the left and right singular vectors of A.
///
/// Note that the routine returns VT $= V^H$, not V.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] jobu
///     Specifies options for computing all or part of the matrix U:
///     - lapack::Job::AllVec:
///         all m columns of U are returned in array U:
///     - lapack::Job::SomeVec:
///         the first min(m,n) columns of U (the left singular vectors)
///         are returned in the array U;
///     - lapack::Job::OverwriteVec:
///         the first min(m,n) columns of U (the left singular vectors)
///         are overwritten on the array A;
///     - lapack::Job::NoVec:
///         no columns of U (no left singular vectors) are computed.
///
/// @param[in] jobvt
///     Specifies options for computing all or part of the matrix
///     $V^H$:
///     - lapack::Job::AllVec:
///         all n rows of $V^H$ are returned in the array VT;
///     - lapack::Job::SomeVec:
///         the first min(m,n) rows of $V^H$ (the right singular vectors)
///         are returned in the array VT;
///     - lapack::Job::OverwriteVec:
///         the first min(m,n) rows of $V^H$ (the right singular vectors)
///         are overwritten on the array A;
///     - lapack::Job::NoVec:
///         no rows of $V^H$ (no right singular vectors) are computed.
///     \n
///     jobvt and jobu cannot both be OverwriteVec.
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
///     On exit:
///     - If jobu = OverwriteVec,
///       A is overwritten with the first min(m,n) columns of U
///       (the left singular vectors, stored columnwise);
///
///     - if jobvt = OverwriteVec,
///       A is overwritten with the first min(m,n) rows of $V^H$
///       (the right singular vectors, stored rowwise);
///
///     - if jobu != OverwriteVec and jobvt != OverwriteVec,
///       the contents of A are destroyed.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @param[out] S
///     The vector S of length min(m,n).
///     The singular values of A, sorted so that S(i) >= S(i+1).
///
/// @param[out] U
///     The m-by-ucol matrix U, stored in an ldu-by-ucol array.
///     - If jobu = AllVec, ucol = m and U contains the m-by-m unitary matrix U;
///
///     - if jobu = SomeVec, ucol = min(m,n) and U contains the first min(m,n)
///       columns of U (the left singular vectors, stored columnwise);
///
///     - if jobu = NoVec or OverwriteVec, U is not referenced.
///
/// @param[in] ldu
///     The leading dimension of the array U. ldu >= 1; if
///     jobu = SomeVec or AllVec, ldu >= m.
///
/// @param[out] VT
///     The vrow-by-n matrix VT, stored in an ldvt-by-n array.
///     - If jobvt = AllVec, vrow = n and VT contains the n-by-n unitary matrix
///       $V^H$;
///
///     - if jobvt = SomeVec, VT contains the first min(m,n) rows of
///       $V^H$ (the right singular vectors, stored rowwise);
///
///     - if jobvt = NoVec or OverwriteVec, VT is not referenced.
///
/// @param[in] ldvt
///     The leading dimension of the array VT. ldvt >= 1;
///     - if jobvt = AllVec, ldvt >= n;
///     - if jobvt = SomeVec, ldvt >= min(m,n).
///
/// @return = 0: successful exit.
/// @return > 0: `lapack::bdsqr` did not converge; return value specifies how
///              many superdiagonals of the intermediate bidiagonal form B
///              did not converge to zero.
///
/// @ingroup gesvd
int64_t gesvd(
    lapack::Job jobu, lapack::Job jobvt, int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda,
    double* S,
    std::complex<double>* U, int64_t ldu,
    std::complex<double>* VT, int64_t ldvt )
{
    char jobu_ = to_char( jobu );
    char jobvt_ = to_char( jobvt );
    lapack_int m_ = to_lapack_int( m );
    lapack_int n_ = to_lapack_int( n );
    lapack_int lda_ = to_lapack_int( lda );
    lapack_int ldu_ = to_lapack_int( ldu );
    lapack_int ldvt_ = to_lapack_int( ldvt );
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_zgesvd(
        &jobu_, &jobvt_, &m_, &n_,
        (lapack_complex_double*) A, &lda_,
        S,
        (lapack_complex_double*) U, &ldu_,
        (lapack_complex_double*) VT, &ldvt_,
        (lapack_complex_double*) qry_work, &ineg_one,
        qry_rwork, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );
    std::vector< double > rwork( (5*min(m,n)) );

    LAPACK_zgesvd(
        &jobu_, &jobvt_, &m_, &n_,
        (lapack_complex_double*) A, &lda_,
        S,
        (lapack_complex_double*) U, &ldu_,
        (lapack_complex_double*) VT, &ldvt_,
        (lapack_complex_double*) &work[0], &lwork_,
        &rwork[0], &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
