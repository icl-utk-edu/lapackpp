// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup rot_aux_grp
void lasr(
    lapack::Side side, lapack::Pivot pivot, lapack::Direction direction,
    int64_t m, int64_t n,
    float const* C,
    float const* S,
    float* A, int64_t lda )
{
    // check for overflow
    if (sizeof( int64_t ) > sizeof( lapack_int )) {
        lapack_error_if( std::abs( m ) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs( n ) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs( lda ) > std::numeric_limits<lapack_int>::max() );
    }
    char side_ = to_char( side );
    char pivot_ = to_char( pivot );
    char direction_ = to_char( direction );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;

    LAPACK_slasr(
        &side_, &pivot_, &direction_, &m_, &n_,
        C, S,
        A, &lda_ );
}

// -----------------------------------------------------------------------------
/// @ingroup rot_aux_grp
void lasr(
    lapack::Side side, lapack::Pivot pivot, lapack::Direction direction,
    int64_t m, int64_t n,
    double const* C,
    double const* S,
    double* A, int64_t lda )
{
    // check for overflow
    if (sizeof( int64_t ) > sizeof( lapack_int )) {
        lapack_error_if( std::abs( m ) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs( n ) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs( lda ) > std::numeric_limits<lapack_int>::max() );
    }
    char side_ = to_char( side );
    char pivot_ = to_char( pivot );
    char direction_ = to_char( direction );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;

    LAPACK_dlasr(
        &side_, &pivot_, &direction_, &m_, &n_,
        C, S,
        A, &lda_ );
}

// -----------------------------------------------------------------------------
/// @ingroup rot_aux_grp
void lasr(
    lapack::Side side, lapack::Pivot pivot, lapack::Direction direction,
    int64_t m, int64_t n,
    float const* C,
    float const* S,
    std::complex<float>* A, int64_t lda )
{
    // check for overflow
    if (sizeof( int64_t ) > sizeof( lapack_int )) {
        lapack_error_if( std::abs( m ) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs( n ) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs( lda ) > std::numeric_limits<lapack_int>::max() );
    }
    char side_ = to_char( side );
    char pivot_ = to_char( pivot );
    char direction_ = to_char( direction );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;

    LAPACK_clasr(
        &side_, &pivot_, &direction_, &m_, &n_,
        C, S,
        (lapack_complex_float*) A, &lda_ );
}

// -----------------------------------------------------------------------------
/// Applies a sequence of real plane rotations to a complex matrix
/// A, from either the left or the right.
///
/// When side = Left, the transformation takes the form
///
///     A := P*A
///
/// and when side = Right, the transformation takes the form
///
///     A := A*P^T
///
/// where P is an orthogonal matrix consisting of a sequence of z plane
/// rotations, with z = m when side = Left and z = n when side = Right,
/// and P^T is the transpose of P.
///
/// When direction = Forward (Forward sequence), then
///
///     P = P(z-1) * ... * P(2) * P(1)
///
/// and when direction = Backward (Backward sequence), then
///
///     P = P(1) * P(2) * ... * P(z-1)
///
/// where P(k) is a plane rotation matrix defined by the 2-by-2 rotation
///
///     R(k) = (  c(k)  s(k) )
///            ( -s(k)  c(k) ).
///
/// When pivot = Variable (Variable pivot), the rotation is performed
/// for the plane (k,k+1), i.e., P(k) has the form
///
///     P(k) = (  1                                            )
///            (       ...                                     )
///            (              1                                )
///            (                   c(k)  s(k)                  )
///            (                  -s(k)  c(k)                  )
///            (                                1              )
///            (                                     ...       )
///            (                                            1  )
///
/// where R(k) appears as a rank-2 modification to the identity matrix in
/// rows and columns k and k+1.
///
/// When pivot = Top (Top pivot), the rotation is performed for the
/// plane (1,k+1), so P(k) has the form
///
///     P(k) = (  c(k)                    s(k)                 )
///            (         1                                     )
///            (              ...                              )
///            (                     1                         )
///            ( -s(k)                    c(k)                 )
///            (                                 1             )
///            (                                      ...      )
///            (                                             1 )
///
/// where R(k) appears in rows and columns 1 and k+1.
///
/// Similarly, when pivot = Bottom (Bottom pivot), the rotation is
/// performed for the plane (k,z), giving P(k) the form
///
///     P(k) = ( 1                                             )
///            (      ...                                      )
///            (             1                                 )
///            (                  c(k)                    s(k) )
///            (                         1                     )
///            (                              ...              )
///            (                                     1         )
///            (                 -s(k)                    c(k) )
///
/// where R(k) appears in rows and columns k and z. The rotations are
/// performed without ever forming P(k) explicitly.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] side
///     Whether the plane rotation matrix P is applied to
///     A on the left or the right.
///     - lapack::Side::Left: Left, compute A := P*A
///     - lapack::Side::Right: Right, compute A:= A*P^T
///
/// @param[in] pivot
///     The plane for which P(k) is a plane rotation
///     matrix.
///     - lapack::Pivot::Variable: Variable pivot, the plane (k,k+1)
///     - lapack::Pivot::Top: Top pivot, the plane (1,k+1)
///     - lapack::Pivot::Bottom: Bottom pivot, the plane (k,z)
///
/// @param[in] direction
///     Whether P is a forward or backward sequence of
///     plane rotations.
///     - lapack::Direction::Forward: Forward, P = P(z-1)*...*P(2)*P(1)
///     - lapack::Direction::Backward: Backward, P = P(1)*P(2)*...*P(z-1)
///
/// @param[in] m
///     The number of rows of the matrix A. If m <= 1, an immediate
///     return is effected.
///
/// @param[in] n
///     The number of columns of the matrix A. If n <= 1, an
///     immediate return is effected.
///
/// @param[in] C
///     The vector C of length m-1 if side = Left; n-1 if side = Right.
///         (m-1) if side = Left
///         (n-1) if side = Right
///     The cosines c(k) of the plane rotations.
///
/// @param[in] S
///     The vector S of length m-1 if side = Left; n-1 if side = Right.
///         (m-1) if side = Left
///         (n-1) if side = Right
///     The sines s(k) of the plane rotations. The 2-by-2 plane
///     rotation part of the matrix P(k), R(k), has the form
///         R(k) = ( c(k)   s(k) )
///                ( -s(k)  c(k) ).
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     The m-by-n matrix A. On exit, A is overwritten by P*A if
///     side = Right or by A*P^T if side = Left.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @ingroup rot_aux_grp
void lasr(
    lapack::Side side, lapack::Pivot pivot, lapack::Direction direction,
    int64_t m, int64_t n,
    double const* C,
    double const* S,
    std::complex<double>* A, int64_t lda )
{
    // check for overflow
    if (sizeof( int64_t ) > sizeof( lapack_int )) {
        lapack_error_if( std::abs( m ) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs( n ) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs( lda ) > std::numeric_limits<lapack_int>::max() );
    }
    char side_ = to_char( side );
    char pivot_ = to_char( pivot );
    char direction_ = to_char( direction );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;

    LAPACK_zlasr(
        &side_, &pivot_, &direction_, &m_, &n_,
        C, S,
        (lapack_complex_double*) A, &lda_ );
}

}  // namespace lapack
