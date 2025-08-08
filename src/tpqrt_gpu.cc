// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/device.hh"

namespace lapack {

using blas::max, blas::min;

// -----------------------------------------------------------------------------
/// Computes a blocked QR factorization of a complex
/// "triangular-pentagonal" matrix C, which is composed of a
/// triangular block A and pentagonal block B, using the compact
/// WY representation for Q.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`
///
/// @param[in] m
///          The number of rows of the matrix B.
///          m >= 0.
///
/// @param[in] n
///          The number of columns of the matrix B, and the order of the
///          triangular matrix A.
///          n >= 0.
///
/// @param[in] l
///          The number of rows of the upper trapezoidal part of B.
///          min(m,n) >= l >= 0.  See Further Details.
///
/// @param[in] nb
///          The block size to be used in the blocked QR.  n >= nb >= 1.
///
/// @param[in,out] dA
///          On entry, the upper triangular n-by-n matrix A.
///          On exit, the elements on and above the diagonal of the array
///          contain the upper triangular matrix R.
///
/// @param[in] ldda
///          The leading dimension of the array A.  ldda >= max(1,n).
///
/// @param[in,out] dB
///          On entry, the pentagonal M-by-N matrix B.  The first M-L rows
///          are rectangular, and the last L rows are upper trapezoidal.
///          On exit, B contains the pentagonal matrix V.  See Further Details.
///
/// @param[in] lddb
///          The leading dimension of the array B.  lddb >= max(1,M).
///
/// @param[out] dT
///          The upper triangular block reflectors stored in compact form
///          as a sequence of upper triangular blocks.  See Further Details.
///
/// @param[in] lddt
///          The leading dimension of the array T.  lddt >= nb.
///
/// @return = 0: successful exit
///
// -----------------------------------------------------------------------------
/// @par Further Details:
///
///
///  The input matrix C is a (N+M)-by-N matrix
/// \[
///               C = \begin{bmatrix}
///                       A
///                   \\  B
///               \end{bmatrix}
/// \]
///
///  where A is an upper triangular N-by-N matrix, and B is M-by-N pentagonal
///  matrix consisting of a (M-L)-by-N rectangular matrix B1 on top of a L-by-N
///  upper trapezoidal matrix B2:
///
///               B = [ B1 ]  <- (M-L)-by-N rectangular
///                   [ B2 ]  <-     L-by-N upper trapezoidal.
///
///  The upper trapezoidal matrix B2 consists of the first L rows of a
///  N-by-N upper triangular matrix, where 0 <= L <= MIN(M,N).  If L=0,
///  B is rectangular M-by-N; if M=L=N, B is upper triangular.
///
///  The matrix W stores the elementary reflectors H(i) in the i-th column
///  below the diagonal (of A) in the (N+M)-by-N input matrix C
///
///               C = [ A ]  <- upper triangular N-by-N
///                   [ B ]  <- M-by-N pentagonal
///
///  so that W can be represented as
///
///               W = [ I ]  <- identity, N-by-N
///                   [ V ]  <- M-by-N, same form as B.
///
///  Thus, all of information needed for W is contained on exit in B, which
///  we call V above.  Note that V has the same form as B; that is,
///
///               V = [ V1 ] <- (M-L)-by-N rectangular
///                   [ V2 ] <-     L-by-N upper trapezoidal.
///
///  The columns of V represent the vectors which define the H(i)'s.
///
///  The number of blocks is B = ceiling(N/NB), where each
///  block is of order NB except for the last block, which is of order
///  IB = N - (B-1)*NB.  For each of the B blocks, a upper triangular block
///  reflector factor is computed: T1, T2, ..., TB.  The NB-by-NB (and IB-by-IB
///  for the last block) T's are stored in the NB-by-N matrix T as
///
///               T = [T1 T2 ... TB].
///

// -----------------------------------------------------------------------------
/// @ingroup tpqrt
template <typename scalar_t>
int64_t tpqrt(
    int64_t m, int64_t n, int64_t l, int64_t nb,
    scalar_t* dA, int64_t ldda,
    scalar_t* dB, int64_t lddb,
    scalar_t* dT, int64_t lddt,
    lapack::Queue& queue )
{
    #define dA(i_, j_) ( dA + i_ + (j_)*ldda )
    #define dB(i_, j_) ( dB + i_ + (j_)*lddb )
    #define dT(i_, j_) ( dT + i_ + (j_)*lddt )

    int64_t info = 0;
    if (m < 0)
        info = -1;
    else if (n < 0)
        info = -2;
    else if (l < 0 || (l > min( m, n ) && min( m, n ) >= 0))
        info = -3;
    else if (nb < 1 || (nb > n && n > 0))
        info = -4;
    else if (ldda < max( 1, n ))
        info = -6;
    else if (lddb < max( 1, m ))
        info = -8;
    else if (lddt < nb)
        info = -10;

    if (info != 0) {
        return info;
    }

    // Quick return if possible
    if (m == 0 || n == 0)
        return info;

    for (int i = 0; i < n; i += nb) {
        // Compute the QR factorization of the current block
        int64_t ib = min( n - i, nb );              // width of block
        int64_t mb = min( m - l + i + ib, m );      // height of block
        int64_t lb = max( mb - (m - l + i), 0 );    // height of trapezoidal part

        tpqrt2( mb, ib, lb, dA(i, i), ldda, dB(0, i), lddb, dT(0, i), lddt, queue );

        // Update by applying H**H to B(:, i+ib:n) from the left
        if (i + ib < n) {
            Op trans = (blas::is_complex_v<scalar_t> ? Op::ConjTrans : Op::Trans);
            tprfb( Side::Left, trans, Direction::Forward, StoreV::Columnwise,
                   mb, n-i-ib, ib, lb, dB(0, i), lddb, dT(0, i), lddt,
                   dA(i, i+ib), ldda, dB(0, i+ib), lddb, queue );
        }
    }

    return info;
}

template int64_t tpqrt(
    int64_t m, int64_t n, int64_t l, int64_t nb,
    float* dA, int64_t ldda,
    float* dB, int64_t lddb,
    float* dT, int64_t lddt,
    lapack::Queue& queue );

template int64_t tpqrt(
    int64_t m, int64_t n, int64_t l, int64_t nb,
    double* dA, int64_t ldda,
    double* dB, int64_t lddb,
    double* dT, int64_t lddt,
    lapack::Queue& queue );

template int64_t tpqrt(
    int64_t m, int64_t n, int64_t l, int64_t nb,
    std::complex<float>* dA, int64_t ldda,
    std::complex<float>* dB, int64_t lddb,
    std::complex<float>* dT, int64_t lddt,
    lapack::Queue& queue );

template int64_t tpqrt(
    int64_t m, int64_t n, int64_t l, int64_t nb,
    std::complex<double>* dA, int64_t ldda,
    std::complex<double>* dB, int64_t lddb,
    std::complex<double>* dT, int64_t lddt,
    lapack::Queue& queue );

}  // namespace lapack
