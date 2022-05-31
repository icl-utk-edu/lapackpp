// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#if LAPACK_VERSION >= 30300  // >= 3.3

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup bbcsd
int64_t bbcsd(
    lapack::Job jobu1, lapack::Job jobu2, lapack::Job jobv1t, lapack::Job jobv2t, lapack::Op trans, int64_t m, int64_t p, int64_t q,
    float* theta,
    float* phi,
    float* U1, int64_t ldu1,
    float* U2, int64_t ldu2,
    float* V1T, int64_t ldv1t,
    float* V2T, int64_t ldv2t,
    float* B11D,
    float* B11E,
    float* B12D,
    float* B12E,
    float* B21D,
    float* B21E,
    float* B22D,
    float* B22E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(q) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu1) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu2) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv1t) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv2t) > std::numeric_limits<lapack_int>::max() );
    }
    char jobu1_ = job_csd2char( jobu1 );
    char jobu2_ = job_csd2char( jobu2 );
    char jobv1t_ = job_csd2char( jobv1t );
    char jobv2t_ = job_csd2char( jobv2t );
    char trans_ = op2char( trans );
    lapack_int m_ = (lapack_int) m;
    lapack_int p_ = (lapack_int) p;
    lapack_int q_ = (lapack_int) q;
    lapack_int ldu1_ = (lapack_int) ldu1;
    lapack_int ldu2_ = (lapack_int) ldu2;
    lapack_int ldv1t_ = (lapack_int) ldv1t;
    lapack_int ldv2t_ = (lapack_int) ldv2t;
    lapack_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_sbbcsd(
        &jobu1_, &jobu2_, &jobv1t_, &jobv2t_, &trans_, &m_, &p_, &q_,
        theta,
        phi,
        U1, &ldu1_,
        U2, &ldu2_,
        V1T, &ldv1t_,
        V2T, &ldv2t_,
        B11D,
        B11E,
        B12D,
        B12E,
        B21D,
        B21E,
        B22D,
        B22E,
        qry_work, &ineg_one, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< float > work( lwork_ );

    LAPACK_sbbcsd(
        &jobu1_, &jobu2_, &jobv1t_, &jobv2t_, &trans_, &m_, &p_, &q_,
        theta,
        phi,
        U1, &ldu1_,
        U2, &ldu2_,
        V1T, &ldv1t_,
        V2T, &ldv2t_,
        B11D,
        B11E,
        B12D,
        B12E,
        B21D,
        B21E,
        B22D,
        B22E,
        &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup bbcsd
int64_t bbcsd(
    lapack::Job jobu1, lapack::Job jobu2, lapack::Job jobv1t, lapack::Job jobv2t, lapack::Op trans, int64_t m, int64_t p, int64_t q,
    double* theta,
    double* phi,
    double* U1, int64_t ldu1,
    double* U2, int64_t ldu2,
    double* V1T, int64_t ldv1t,
    double* V2T, int64_t ldv2t,
    double* B11D,
    double* B11E,
    double* B12D,
    double* B12E,
    double* B21D,
    double* B21E,
    double* B22D,
    double* B22E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(q) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu1) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu2) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv1t) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv2t) > std::numeric_limits<lapack_int>::max() );
    }
    char jobu1_ = job_csd2char( jobu1 );
    char jobu2_ = job_csd2char( jobu2 );
    char jobv1t_ = job_csd2char( jobv1t );
    char jobv2t_ = job_csd2char( jobv2t );
    char trans_ = op2char( trans );
    lapack_int m_ = (lapack_int) m;
    lapack_int p_ = (lapack_int) p;
    lapack_int q_ = (lapack_int) q;
    lapack_int ldu1_ = (lapack_int) ldu1;
    lapack_int ldu2_ = (lapack_int) ldu2;
    lapack_int ldv1t_ = (lapack_int) ldv1t;
    lapack_int ldv2t_ = (lapack_int) ldv2t;
    lapack_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_dbbcsd(
        &jobu1_, &jobu2_, &jobv1t_, &jobv2t_, &trans_, &m_, &p_, &q_,
        theta,
        phi,
        U1, &ldu1_,
        U2, &ldu2_,
        V1T, &ldv1t_,
        V2T, &ldv2t_,
        B11D,
        B11E,
        B12D,
        B12E,
        B21D,
        B21E,
        B22D,
        B22E,
        qry_work, &ineg_one, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< double > work( lwork_ );

    LAPACK_dbbcsd(
        &jobu1_, &jobu2_, &jobv1t_, &jobv2t_, &trans_, &m_, &p_, &q_,
        theta,
        phi,
        U1, &ldu1_,
        U2, &ldu2_,
        V1T, &ldv1t_,
        V2T, &ldv2t_,
        B11D,
        B11E,
        B12D,
        B12E,
        B21D,
        B21E,
        B22D,
        B22E,
        &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup bbcsd
int64_t bbcsd(
    lapack::Job jobu1, lapack::Job jobu2, lapack::Job jobv1t, lapack::Job jobv2t, lapack::Op trans, int64_t m, int64_t p, int64_t q,
    float* theta,
    float* phi,
    std::complex<float>* U1, int64_t ldu1,
    std::complex<float>* U2, int64_t ldu2,
    std::complex<float>* V1T, int64_t ldv1t,
    std::complex<float>* V2T, int64_t ldv2t,
    float* B11D,
    float* B11E,
    float* B12D,
    float* B12E,
    float* B21D,
    float* B21E,
    float* B22D,
    float* B22E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(q) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu1) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu2) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv1t) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv2t) > std::numeric_limits<lapack_int>::max() );
    }
    char jobu1_ = job_csd2char( jobu1 );
    char jobu2_ = job_csd2char( jobu2 );
    char jobv1t_ = job_csd2char( jobv1t );
    char jobv2t_ = job_csd2char( jobv2t );
    char trans_ = op2char( trans );
    lapack_int m_ = (lapack_int) m;
    lapack_int p_ = (lapack_int) p;
    lapack_int q_ = (lapack_int) q;
    lapack_int ldu1_ = (lapack_int) ldu1;
    lapack_int ldu2_ = (lapack_int) ldu2;
    lapack_int ldv1t_ = (lapack_int) ldv1t;
    lapack_int ldv2t_ = (lapack_int) ldv2t;
    lapack_int info_ = 0;

    // query for workspace size
    float qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_cbbcsd(
        &jobu1_, &jobu2_, &jobv1t_, &jobv2t_, &trans_, &m_, &p_, &q_,
        theta,
        phi,
        (lapack_complex_float*) U1, &ldu1_,
        (lapack_complex_float*) U2, &ldu2_,
        (lapack_complex_float*) V1T, &ldv1t_,
        (lapack_complex_float*) V2T, &ldv2t_,
        B11D,
        B11E,
        B12D,
        B12E,
        B21D,
        B21E,
        B22D,
        B22E,
        qry_rwork, &ineg_one, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lrwork_ = real(qry_rwork[0]);

    // allocate workspace
    lapack::vector< float > rwork( lrwork_ );

    LAPACK_cbbcsd(
        &jobu1_, &jobu2_, &jobv1t_, &jobv2t_, &trans_, &m_, &p_, &q_,
        theta,
        phi,
        (lapack_complex_float*) U1, &ldu1_,
        (lapack_complex_float*) U2, &ldu2_,
        (lapack_complex_float*) V1T, &ldv1t_,
        (lapack_complex_float*) V2T, &ldv2t_,
        B11D,
        B11E,
        B12D,
        B12E,
        B21D,
        B21E,
        B22D,
        B22E,
        &rwork[0], &lrwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the CS decomposition of a unitary matrix in
/// bidiagonal-block form,
/// \[
///     X =
///     \begin{bmatrix}
///             B_{11}  &  B_{12}  &   0  &  0
///         \\  0       &  0       &  -I  &  0
///         \\  \hline
///             B_{21}  &  B_{22}  &   0  &  0
///         \\  0       &  0       &   0  &  I
///     \end{bmatrix}
///     =
///     \begin{bmatrix}
///             U_{1}  &
///         \\  \hline
///                    &  U_{2}
///     \end{bmatrix}
///     \begin{bmatrix}
///             C  &  -S  &   0  &  0
///         \\  0  &   0  &  -I  &  0
///         \\  \hline
///             S  &   C  &   0  &  0
///         \\  0  &   0  &   0  &  I
///     \end{bmatrix}
///     \begin{bmatrix}
///             V_{1}  &
///         \\  \hline
///                    &  V_{2}
///     \end{bmatrix}^H
/// \]
///
/// X is m-by-m, its top-left block is p-by-q, and q must be no larger
/// than p, m-p, or m-q. (If q is not the smallest index, then X must be
/// transposed and/or permuted. This can be done in constant time using
/// the trans and signs options. See `lapack::uncsd` for details.)
///
/// The bidiagonal matrices B11, B12, B21, and B22 are represented
/// implicitly by angles theta(1:q) and phi(1:q-1).
///
/// The unitary matrices U1, U2, V1T, and V2T are input/output.
/// The input matrices are pre- or post-multiplied by the appropriate
/// singular vector matrices.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] jobu1
///     - lapack::Job::UpdateVec: U1 is updated;
///     - lapack::Job::NoVec:     U1 is not updated.
///
/// @param[in] jobu2
///     - lapack::Job::UpdateVec: U2 is updated;
///     - lapack::Job::NoVec:     U2 is not updated.
///
/// @param[in] jobv1t
///     - lapack::Job::UpdateVec: V1T is updated;
///     - lapack::Job::NoVec:     V1T is not updated.
///
/// @param[in] jobv2t
///     - lapack::Job::UpdateVec: V2T is updated;
///     - lapack::Job::NoVec:     V2T is not updated.
///
/// @param[in] trans
///     - lapack::Op::Trans:
///             X, U1, U2, V1T, and V2T are stored in row-major order;
///     - lapack::Op::NoTrans:
///             X, U1, U2, V1T, and V2T are stored in column-major order.
///
/// @param[in] m
///     The number of rows and columns in X, the unitary matrix in
///     bidiagonal-block form.
///
/// @param[in] p
///     The number of rows in the top-left block of X. 0 <= p <= m.
///
/// @param[in] q
///     The number of columns in the top-left block of X.
///     0 <= q <= min(p, m-p, m-q).
///
/// @param[in,out] theta
///     The vector theta of length q.
///     On entry, the angles theta(1), ..., theta(q) that, along with
///     phi(1), ..., phi(q-1), define the matrix in bidiagonal-block
///     form. On exit, the angles whose cosines and sines define the
///     diagonal blocks in the CS decomposition.
///
/// @param[in,out] phi
///     The vector phi of length q-1.
///     The angles phi(1), ..., phi(q-1) that, along with
///     theta(1), ..., theta(q), define the matrix in bidiagonal-block form.
///
/// @param[in,out] U1
///     The p-by-p matrix U1, stored in an ldu1-by-p array.
///     On entry, a p-by-p matrix. On exit, U1 is postmultiplied
///     by the left singular vector matrix common to [ B11 ; 0 ] and
///     [ B12 0 0 ; 0 -I 0 0 ].
///
/// @param[in] ldu1
///     The leading dimension of the array U1, ldu1 >= max(1,p).
///
/// @param[in,out] U2
///     The (m-p)-by-(m-p) matrix U2, stored in an ldu2-by-(m-p) array.
///     On entry, an (m-p)-by-(m-p) matrix. On exit, U2 is
///     postmultiplied by the left singular vector matrix common to
///     [ B21 ; 0 ] and [ B22 0 0 ; 0 0 I ].
///
/// @param[in] ldu2
///     The leading dimension of the array U2, ldu2 >= max(1,m-p).
///
/// @param[in,out] V1T
///     The q-by-q matrix V1T, stored in an ldv1t-by-q array.
///     On entry, a q-by-q matrix. On exit, V1T is premultiplied
///     by the conjugate transpose of the right singular vector
///     matrix common to [ B11 ; 0 ] and [ B21 ; 0 ].
///
/// @param[in] ldv1t
///     The leading dimension of the array V1T, ldv1t >= max(1,q).
///
/// @param[in,out] V2T
///     The (m-q)-by-(m-q) matrix V2T, stored in an ldv2t-by-(m-q) array.
///     On entry, an (m-q)-by-(m-q) matrix. On exit, V2T is
///     premultiplied by the conjugate transpose of the right
///     singular vector matrix common to [ B12 0 0 ; 0 -I 0 ] and
///     [ B22 0 0 ; 0 0 I ].
///
/// @param[in] ldv2t
///     The leading dimension of the array V2T, ldv2t >= max(1,m-q).
///
/// @param[out] B11D
///     The vector B11D of length q.
///     When bbcsd converges, B11D contains the cosines of
///     theta(1), ..., theta(q). If `bbcsd` fails to converge, then B11D
///     contains the diagonal of the partially reduced top-left
///     block.
///
/// @param[out] B11E
///     The vector B11E of length q-1.
///     When `bbcsd` converges, B11E contains zeros. If `bbcsd` fails
///     to converge, then B11E contains the superdiagonal of the
///     partially reduced top-left block.
///
/// @param[out] B12D
///     The vector B12D of length q.
///     When `bbcsd` converges, B12D contains the negative sines of
///     theta(1), ..., theta(q). If `bbcsd` fails to converge, then
///     B12D contains the diagonal of the partially reduced top-right
///     block.
///
/// @param[out] B12E
///     The vector B12E of length q-1.
///     When `bbcsd` converges, B12E contains zeros. If `bbcsd` fails
///     to converge, then B12E contains the subdiagonal of the
///     partially reduced top-right block.
///
/// @param[out] B21D
///     The vector B21D of length q.
///     When `bbcsd` converges, B21D contains the negative sines of
///     theta(1), ..., theta(q). If `bbcsd` fails to converge, then
///     B21D contains the diagonal of the partially reduced bottom-left
///     block.
///
/// @param[out] B21E
///     The vector B21E of length q-1.
///     When `bbcsd` converges, B21E contains zeros. If `bbcsd` fails
///     to converge, then B21E contains the subdiagonal of the
///     partially reduced bottom-left block.
///
/// @param[out] B22D
///     The vector B22D of length q.
///     When `bbcsd` converges, B22D contains the negative sines of
///     theta(1), ..., theta(q). If `bbcsd` fails to converge, then
///     B22D contains the diagonal of the partially reduced bottom-right
///     block.
///
/// @param[out] B22E
///     The vector B22E of length q-1.
///     When `bbcsd` converges, B22E contains zeros. If `bbcsd` fails
///     to converge, then B22E contains the subdiagonal of the
///     partially reduced bottom-right block.
///
/// @return = 0: successful exit.
/// @return > 0: did not converge; return value specifies the number
///              of nonzero entries in phi, and B11D, B11E, etc.,
///              contain the partially reduced matrix.
///
/// @ingroup bbcsd
int64_t bbcsd(
    lapack::Job jobu1, lapack::Job jobu2, lapack::Job jobv1t, lapack::Job jobv2t, lapack::Op trans, int64_t m, int64_t p, int64_t q,
    double* theta,
    double* phi,
    std::complex<double>* U1, int64_t ldu1,
    std::complex<double>* U2, int64_t ldu2,
    std::complex<double>* V1T, int64_t ldv1t,
    std::complex<double>* V2T, int64_t ldv2t,
    double* B11D,
    double* B11E,
    double* B12D,
    double* B12E,
    double* B21D,
    double* B21E,
    double* B22D,
    double* B22E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(p) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(q) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu1) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldu2) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv1t) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldv2t) > std::numeric_limits<lapack_int>::max() );
    }
    char jobu1_ = job_csd2char( jobu1 );
    char jobu2_ = job_csd2char( jobu2 );
    char jobv1t_ = job_csd2char( jobv1t );
    char jobv2t_ = job_csd2char( jobv2t );
    char trans_ = op2char( trans );
    lapack_int m_ = (lapack_int) m;
    lapack_int p_ = (lapack_int) p;
    lapack_int q_ = (lapack_int) q;
    lapack_int ldu1_ = (lapack_int) ldu1;
    lapack_int ldu2_ = (lapack_int) ldu2;
    lapack_int ldv1t_ = (lapack_int) ldv1t;
    lapack_int ldv2t_ = (lapack_int) ldv2t;
    lapack_int info_ = 0;

    // query for workspace size
    double qry_rwork[1];
    lapack_int ineg_one = -1;
    LAPACK_zbbcsd(
        &jobu1_, &jobu2_, &jobv1t_, &jobv2t_, &trans_, &m_, &p_, &q_,
        theta,
        phi,
        (lapack_complex_double*) U1, &ldu1_,
        (lapack_complex_double*) U2, &ldu2_,
        (lapack_complex_double*) V1T, &ldv1t_,
        (lapack_complex_double*) V2T, &ldv2t_,
        B11D,
        B11E,
        B12D,
        B12E,
        B21D,
        B21E,
        B22D,
        B22E,
        qry_rwork, &ineg_one, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lrwork_ = real(qry_rwork[0]);

    // allocate workspace
    lapack::vector< double > rwork( lrwork_ );

    LAPACK_zbbcsd(
        &jobu1_, &jobu2_, &jobv1t_, &jobv2t_, &trans_, &m_, &p_, &q_,
        theta,
        phi,
        (lapack_complex_double*) U1, &ldu1_,
        (lapack_complex_double*) U2, &ldu2_,
        (lapack_complex_double*) V1T, &ldv1t_,
        (lapack_complex_double*) V2T, &ldv2t_,
        B11D,
        B11E,
        B12D,
        B12E,
        B21D,
        B21E,
        B22D,
        B22E,
        &rwork[0], &lrwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1, 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.3.0
