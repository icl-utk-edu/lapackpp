// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"
#include "lapack/device.hh"
#include "NoConstructAllocator.hh"

namespace lapack
{

using blas::Uplo, blas::Op;
using std::min, std::max;

// -----------------------------------------------------------------------------
/// @ingroup tpqrt
template <typename scalar_t>
void tprfb(
    lapack::Side side, lapack::Op trans,
    lapack::Direction direction, lapack::StoreV storev,
    int64_t m, int64_t n, int64_t k, int64_t l,
    scalar_t const* dV, int64_t lddv,
    scalar_t const* dT, int64_t lddt,
    scalar_t* dA, int64_t ldda,
    scalar_t* dB, int64_t lddb,
    lapack::Queue& queue )
{
    #define dA(i_, j_) ( dA + i_ + (j_)*ldda )
    #define dB(i_, j_) ( dB + i_ + (j_)*lddb )
    #define dV(i_, j_) ( dV + i_ + (j_)*lddv )

    scalar_t zero = 0.0;
    scalar_t one  = 1.0;

    // Quick return if possible

    if (m <= 0 || n <= 0 || k <= 0 || l < 0) {
        return;
    }

    bool column, row;
    bool left, right;
    bool forward, backward;

    if (storev == StoreV::Columnwise)
        column = true;
    else if (storev == StoreV::Rowwise)
        column = false;
    else
        return;
    row = !column;

    if (side == Side::Left)
        left = true;
    else if (side == Side::Right)
        left = false;
    else
        return;
    right = !left;

    if (direction == Direction::Forward)
        forward = true;
    else if (direction == Direction::Backward)
        forward = false;
    else
        return;
    backward = !forward;

    // Allocate workspace
    int64_t ldwork = (left ? k : m);
    int64_t lwork = (left ? k*n : m*k);
    scalar_t* work = blas::device_malloc< scalar_t >( lwork, queue );
    queue.sync();

    #define work(i_, j_) ( work + i_ + j_*ldwork )

    // ConjTrans for complex, Trans for real
    Op op_trans = Op::Trans;
    if (blas::is_complex_v<scalar_t>) {
        op_trans = Op::ConjTrans;
    }

    // Offsets for matrix blocks
    int64_t mp, np, kp;

    // Pointers for matrix blocks
    // B matrix
    scalar_t* B_1;      // Top or left block of B
    scalar_t* B_2;      // Bottom or right block of B

    // V matrix; Not all V blocks are used in every case
    scalar_t const* V_11;     // Top-left block of V
    scalar_t const* V_12;     // Top-right block of V
    scalar_t const* V_21;     // Bottom-left block of V
    scalar_t const* V_22;     // Bottom-right block of V
    scalar_t const* V_1x;     // Top block of V (V_1x = [ V_11 V_12 ])
    scalar_t const* V_2x;     // Bottom block of V (V_2x = [ V_21 V_22 ])
    scalar_t const* V_x1;     // Left block of V (V_x1 = [ V_11; V_21 ])
    scalar_t const* V_x2;     // Right block of V (V_x2 = [ V_12; V_22 ])

    // Work matrix
    scalar_t* work_1;   // Top or left block of work
    scalar_t* work_2;   // Bottom or right block of work

    // ---------------------------------------------------------------------------

    if (column && forward && left) {

        // ---------------------------------------------------------------------------
        //
        //        Let  W =  [ I ]    (k-by-k)
        //                  [ V ]    (m-by-k)
        //
        //        Form  op(H) C  where  C = [ A ]  (k-by-n)
        //                                  [ B ]  (m-by-n)
        //
        //        H = I - W op(T) W^H
        //
        //        A = A -   op(T) (A + V^H B)
        //        B = B - V op(T) (A + V^H B)
        //
        // ---------------------------------------------------------------------------
        //
        //                 n
        //        B    = [ B_1 ] m-l
        //               [ B_2 ] l
        //
        //                 l     k-l
        //        V    = [ V_11  V_12 ] m-l
        //               [ V_21  V_22 ] l
        //
        //                 n
        //        work = [ work_1 ] l
        //               [ work_2 ] k-l
        //
        // ---------------------------------------------------------------------------

        // mp = offset of bottom of B and V
        mp = min(m - l, m);
        // kp = offset of bottom of work and right of V
        kp = min(l, k);

        B_1 = dB;
        B_2 = dB(mp, 0);

        V_11 = dV;
        V_21 = dV(mp, 0);
        V_12 = dV(0, kp);
        V_22 = dV(mp, kp);
        V_1x = V_11;
        V_2x = V_21;
        V_x1 = V_11;
        V_x2 = V_12;

        work_1 = work;
        work_2 = work(kp, 0);

        // work = V^H B
        // work_1 = B_2
        blas::device_memcpy_2d( work_1, ldwork, B_2, lddb, l, n, queue );
        // work_1 = V_21^H work_1 = V_21^H B_2
        blas::trmm( Layout::ColMajor, Side::Left, Uplo::Upper, op_trans, Diag::NonUnit, l, n, one, V_21, lddv, work_1, ldwork, queue );
        // work_1 = V_11^H B_1 + work_1
        blas::gemm( Layout::ColMajor, op_trans, Op::NoTrans, l, n, m - l, one, V_11, lddv, B_1, lddb, one, work_1, ldwork, queue );
        // work_2 = V_x2^H B + work_2
        blas::gemm( Layout::ColMajor, op_trans, Op::NoTrans, k - l, n, m, one, V_x2, lddv, dB, lddb, zero, work_2, ldwork, queue );

        // work = A + V^H B
        // work += A
        blas::geadd( Layout::ColMajor, Op::NoTrans, k, n, one, dA, ldda, one, work, ldwork, queue );

        // work = op(T) (A + V^H B)
        // work = op(T) work
        blas::trmm( Layout::ColMajor, Side::Left, Uplo::Upper, trans, Diag::NonUnit, k, n, one, dT, lddt, work, ldwork, queue );

        // A = A - op(T) (A + V^H B)
        // A = A - work
        blas::geadd( Layout::ColMajor, Op::NoTrans, k, n, -one, work, ldwork, one, dA, ldda, queue );

        // B = B - V op(T) (A + V^H B) = B - V work
        // B_1 = -V_1x work + B_1
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, m - l, n, k, -one, V_1x, lddv, work, ldwork, one, B_1, lddb, queue );
        // B_2 = -V_22 work_2 + B_2
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, l, n, k - l, -one, V_22, lddv, work_2, ldwork, one, B_2, lddb, queue );
        // work_1 = V_21 work_1
        blas::trmm( Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, l, n, -one, V_21, lddv, work_1, ldwork, queue );
        // B_2 = B_2 - work_1
        blas::geadd( Layout::ColMajor, Op::NoTrans, l, n, one, work_1, ldwork, one, B_2, lddb, queue );

        // ---------------------------------------------------------------------------
    }
    else if (column && forward && right) {

        // ---------------------------------------------------------------------------
        //
        //        Let  W =  [ I ]    (k-by-k)
        //                  [ V ]    (n-by-k)
        //
        //        Form  C op(H)  where  C = [ A B ] (A is m-by-k, B is m-by-n)
        //
        //        H = I - W op(T) W^H
        //
        //        A = A - (A + B V) op(T)
        //        B = B - (A + B V) op(T) V^H
        //
        // ---------------------------------------------------------------------------
        //
        //                 n-l    l
        //        B    = [ B_1    B_2 ] m
        //
        //                 l      k-l
        //        V    = [ V_11   V_12 ] n-l
        //               [ V_21   V_22 ] l
        //
        //                 l      k-l
        //        work = [ work_1 work2 ] m
        //
        // ---------------------------------------------------------------------------

        // np = offset of right of B and bottom of V
        np = min(n - l, n);
        // kp = offset of right of work and V
        kp = min(l, k);

        B_1 = dB;
        B_2 = dB(0, np);

        V_11 = dV;
        V_21 = dV(np, 0);
        V_12 = dV(0, kp);
        V_22 = dV(np, kp);
        V_1x = V_11;
        V_2x = V_21;
        V_x1 = V_11;
        V_x2 = V_12;

        work_1 = work;
        work_2 = work(0, kp);

        // work = B V
        // work_1 = B_2
        blas::device_memcpy_2d( work_1, ldwork, B_2, lddb, m, l, queue );
        // work_1 = work_1 V_21 = B_2 V_21
        blas::trmm( Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, l, one, V_21, lddv, work_1, ldwork, queue );
        // work_1 = B_1 V_11 + work_1
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, l, n - l, one, B_1, lddb, V_11, lddv, one, work_1, ldwork, queue );
        // work_2 = B V_x2 + work_2
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k - l, n, one, dB, lddb, V_x2, lddv, zero, work_2, ldwork, queue );

        // work = A + B V
        // work = work + A
        blas::geadd( Layout::ColMajor, Op::NoTrans, m, k, one, dA, ldda, one, work, ldwork, queue );

        // work = (A + B V) op(T)
        // work = work op(T)
        blas::trmm( Layout::ColMajor, Side::Right, Uplo::Upper, trans, Diag::NonUnit, m, k, one, dT, lddt, work, ldwork, queue );

        // A = A - (A + B V) op(T)
        // A = A - work
        blas::geadd( Layout::ColMajor, Op::NoTrans, m, k, -one, work, ldwork, one, dA, ldda, queue );

        // B = B - (A + B V) op(T) V^H = B - work V^H
        // B_1 = -work V_1x^H + B_1
        blas::gemm( Layout::ColMajor, Op::NoTrans, op_trans, m, n - l, k, -one, work, ldwork, V_1x, lddv, one, B_1, lddb, queue );
        // B_2 = -work_2 V_22^H + B_2
        blas::gemm( Layout::ColMajor, Op::NoTrans, op_trans, m, l, k - l, -one, work_2, ldwork, V_22, lddv, one, B_2, lddb, queue );
        // work_1 = work_1 V_21^H
        blas::trmm( Layout::ColMajor, Side::Right, Uplo::Upper, op_trans, Diag::NonUnit, m, l, one, V_21, lddv, work_1, ldwork, queue );
        // B_2 = B_2 - work_1
        blas::geadd( Layout::ColMajor, Op::NoTrans, m, l, -one, work_1, ldwork, one, B_2, lddb, queue );

        // ---------------------------------------------------------------------------
    }
    else if (column && backward && left) {

        // ---------------------------------------------------------------------------
        //
        //        Let  W =  [ V ]    (m-by-k)
        //                  [ I ]    (k-by-k)
        //
        //        Form  op(H) C  where  C = [ B ]  (m-by-n)
        //                                  [ A ]  (k-by-n)
        //
        //        H = I - W op(T) W^H
        //
        //        A = A -   op(T) (A + V^H B)
        //        B = B - V op(T) (A + V^H B)
        //
        // ---------------------------------------------------------------------------
        //
        //                 n
        //        B    = [ B_1 ] l
        //               [ B_2 ] m-l
        //
        //                 k-l   l
        //        V    = [ V_11  V_12 ] l
        //               [ V_21  V_22 ] m-l
        //
        //                 n
        //        work = [ work_1 ] k-l
        //               [ work_2 ] l
        //
        // ---------------------------------------------------------------------------

        // mp = offset of bottom of B and V
        mp = min(l, m);
        // kp = offset of bottom of work and right of V
        kp = min(k - l, k);

        B_1 = dB;
        B_2 = dB(mp, 0);

        V_11 = dV;
        V_21 = dV(mp, 0);
        V_12 = dV(0, kp);
        V_22 = dV(mp, kp);
        V_1x = V_11;
        V_2x = V_21;
        V_x1 = V_11;
        V_x2 = V_12;

        work_1 = work;
        work_2 = work(kp, 0);

        // work = V^H B
        // work_2 = B_1
        blas::device_memcpy_2d( work_2, ldwork, B_1, lddb, l, n, queue );
        // work_2 = V_12^H work_2 = V_12^H B_1
        blas::trmm( Layout::ColMajor, Side::Left, Uplo::Lower, op_trans, Diag::NonUnit, l, n, one, V_12, lddv, work_2, ldwork, queue );
        // work_2 = V_22^H B_2 + work_2
        blas::gemm( Layout::ColMajor, op_trans, Op::NoTrans, l, n, m - l, one, V_22, lddv, B_1, lddb, one, work_2, ldwork, queue );
        // work_1 = V_x1^H B + work_1
        blas::gemm( Layout::ColMajor, op_trans, Op::NoTrans, k - l, n, m, one, V_x1, lddv, dB, lddb, zero, work_1, ldwork, queue );

        // work = A + V^H B
        // work = work + A
        blas::geadd( Layout::ColMajor, Op::NoTrans, k, n, one, dA, ldda, one, work, ldwork, queue );

        // work = op(T) (A + V^H B)
        // work = op(T) work
        blas::trmm( Layout::ColMajor, Side::Left, Uplo::Lower, trans, Diag::NonUnit, k, n, one, dT, lddt, work, ldwork, queue );

        // A = A - op(T) (A + V^H B)
        // A = A - work
        blas::geadd( Layout::ColMajor, Op::NoTrans, k, n, -one, work, ldwork, one, dA, ldda, queue );

        // B = B - V op(T) (A + V^H B) = B - V work
        // B_2 = -V_2x work + B_2
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, m - l, n, k, -one, V_2x, lddv, work, ldwork, one, B_2, lddb, queue );
        // B_1 = -V_11 work_1 + B_1
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, l, n, k - l, -one, V_11, lddv, work_1, ldwork, one, B_1, lddb, queue );
        // work_2 = V_12 work_2
        blas::trmm( Layout::ColMajor, Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit, l, n, one, V_12, lddv, work_2, ldwork, queue );
        // B_1 = B_1 - work_2
        blas::geadd( Layout::ColMajor, Op::NoTrans, l, n, -one, work_2, ldwork, one, B_1, lddb, queue );

        // ---------------------------------------------------------------------------
    }
    else if (column && backward && right) {

        // ---------------------------------------------------------------------------
        //
        //        Let  W =  [ V ]    (n-by-k)
        //                  [ I ]    (k-by-k)
        //
        //        Form  C op(H)  where  C = [ B A ] (B is m-by-n, A is m-by-k)
        //
        //        H = I - W op(T) W^H
        //
        //        A = A - (A + B V) op(T)
        //        B = B - (A + B V) op(T) V^H
        //
        // ---------------------------------------------------------------------------
        //
        //                 l      n-l
        //        B    = [ B_1    B_2 ] m
        //
        //                 k-l    l
        //        V    = [ V_11   V_12 ] l
        //               [ V_21   V_22 ] n-l
        //
        //                 k-l    l
        //        work = [ work_1 work2 ] m
        //
        // ---------------------------------------------------------------------------

        // np = offset of right of B and bottom of V
        np = min(l, n);
        // kp = offset of right of work and V
        kp = min(k - l, k);

        B_1 = dB;
        B_2 = dB(0, np);

        V_11 = dV;
        V_21 = dV(np, 0);
        V_12 = dV(0, kp);
        V_22 = dV(np, kp);
        V_1x = V_11;
        V_2x = V_21;
        V_x1 = V_11;
        V_x2 = V_12;

        work_1 = work;
        work_2 = work(0, kp);

        // work = B V
        // work_2 = B_1
        blas::device_memcpy_2d( work_2, ldwork, B_1, lddb, m, l, queue );
        // work_2 = work_2 V_12 = B_1 V_12
        blas::trmm( Layout::ColMajor, Side::Right, Uplo::Lower, Op::NoTrans, Diag::NonUnit, m, l, one, V_12, lddv, work_2, ldwork, queue );
        // work_2 = B2 V_22 + work_2
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, l, n - l, one, B_1, lddb, V_22, lddv, one, work_2, ldwork, queue );
        // work_1 = B V_x1 + work_1
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k - l, n, one, dB, lddb, V_x1, lddv, zero, work_1, ldwork, queue );

        // work = A + B V
        // work = work + A
        blas::geadd( Layout::ColMajor, Op::NoTrans, m, k, one, dA, ldda, one, work, ldwork, queue );

        // work = (A + B V) op(T)
        // work = work op(T)
        blas::trmm( Layout::ColMajor, Side::Right, Uplo::Lower, trans, Diag::NonUnit, m, k, one, dT, lddt, work, ldwork, queue );

        // A = A - (A + B V) op(T)
        // A = A - work
        blas::geadd( Layout::ColMajor, Op::NoTrans, m, k, -one, work, ldwork, one, dA, ldda, queue );

        // B = B - (A + B V) op(T) V^H = B - work V^H
        // B_2 = -work V_2x^H + B_2
        blas::gemm( Layout::ColMajor, Op::NoTrans, op_trans, m, n - l, k, -one, work, ldwork, V_2x, lddv, one, B_2, lddb, queue );
        // B_1 = -work_1 V_11^H + B_1
        blas::gemm( Layout::ColMajor, Op::NoTrans, op_trans, m, l, k - l, -one, work_1, ldwork, V_11, lddv, one, B_1, lddb, queue );
        // work_2 = work_2 V_12^H
        blas::trmm( Layout::ColMajor, Side::Right, Uplo::Lower, op_trans, Diag::NonUnit, m, l, one, V_12, lddv, work_2, ldwork, queue );
        // B_1 = B_1 - work_2
        blas::geadd( Layout::ColMajor, Op::NoTrans, m, l, -one, work_2, ldwork, one, B_1, lddb, queue );

        // ---------------------------------------------------------------------------
    }
    else if (row && forward && left) {

        // ---------------------------------------------------------------------------
        //
        //        Let  W =  [ I V ] ( I is k-by-k, V is k-by-m )
        //
        //        Form  op(H) C  where  C = [ A ]  (k-by-n)
        //                                  [ B ]  (m-by-n)
        //
        //        H = I - W^H op(T) W
        //
        //        A = A -     op(T) (A + V B)
        //        B = B - V^H op(T) (A + V B)
        //
        // ---------------------------------------------------------------------------
        //
        //                 n
        //        B    = [ B_1 ] m-l
        //               [ B_2 ] l
        //
        //                 m-l   l
        //        V    = [ V_11  V_12 ] l
        //               [ V_21  V_22 ] k-l
        //
        //                 n
        //        work = [ work_1 ] l
        //               [ work_2 ] k-l
        //
        // ---------------------------------------------------------------------------

        // mp = offset of bottom of B and right of V
        mp = min(m - l, m);
        // kp = offset of bottom of work and V
        kp = min(l, k);

        B_1 = dB;
        B_2 = dB(mp, 0);

        V_11 = dV;
        V_21 = dV(kp, 0);
        V_12 = dV(0, mp);
        V_22 = dV(kp, mp);
        V_1x = V_11;
        V_2x = V_21;
        V_x1 = V_11;
        V_x2 = V_12;

        work_1 = work;
        work_2 = work(kp, 0);

        // work = V B
        // work_1 = B_2
        blas::device_memcpy_2d( work_1, ldwork, B_2, lddb, l, n, queue );
        // work_1 = V_12 work_1 = V_12 B_2
        blas::trmm( Layout::ColMajor, Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit, l, n, one, V_12, lddv, work_1, lddb, queue );
        // work_1 = V_11 B_1 + work_1
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, l, n, m - l, one, V_11, lddv, B_1, lddb, one, work_1, ldwork, queue );
        // work_2 = V_2x B + work_2
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, k - l, n, m, one, V_2x, lddv, dB, lddb, zero, work_2, ldwork, queue );

        // work = A + V B
        // work = work + A
        blas::geadd( Layout::ColMajor, Op::NoTrans, k, n, one, dA, ldda, one, work, ldwork, queue );

        // work = op(T) (A + B V)
        // work = op(T) work
        blas::trmm( Layout::ColMajor, Side::Left, Uplo::Upper, trans, Diag::NonUnit, k, n, one, dT, lddt, work, ldwork, queue );

        // A = A - op(T) (A + B V)
        // A = A - work
        blas::geadd( Layout::ColMajor, Op::NoTrans, k, n, -one, work, ldwork, one, dA, ldda, queue );

        // B = B - V^H op(T) (A + B V) = B - V^H work
        // B_1 = -V_x1^H work + B_1
        blas::gemm( Layout::ColMajor, op_trans, Op::NoTrans, m - l, n, k, -one, V_x1, lddv, work, ldwork, one, B_1, lddb, queue );
        // B_2 = -V_22^H work_2 + B_2
        blas::gemm( Layout::ColMajor, op_trans, Op::NoTrans, l, n, k - l, -one, V_22, lddv, work_2, ldwork, one, B_2, lddb, queue );
        // work_1 = V_12^H work_1
        blas::trmm( Layout::ColMajor, Side::Left, Uplo::Lower, op_trans, Diag::NonUnit, l, n, one, V_12, lddv, work_1, ldwork, queue );
        // B_2 = B_2 - work_1
        blas::geadd( Layout::ColMajor, Op::NoTrans, l, n, -one, work_1, ldwork, one, B_2, lddb, queue );

        // ---------------------------------------------------------------------------
    }
    else if (row && forward && right) {

        // ---------------------------------------------------------------------------
        //
        //        Let  W =  [ I V ] ( I is k-by-k, V is k-by-n )
        //
        //        Form  C op(H)  where  C = [ A B ] (A is m-by-k, B is m-by-n)
        //
        //        H = I - W^H op(T) W
        //
        //        A = A - (A + B V^H) op(T)
        //        B = B - (A + B V^H) op(T) V
        //
        // ---------------------------------------------------------------------------
        //
        //                 n-l    l
        //        B    = [ B_1    B_2 ] m
        //
        //                 n-l    l
        //        V    = [ V_11   V_12 ] l
        //               [ V_21   V_22 ] k-l
        //
        //                 l      k-l
        //        work = [ work_1 work2 ] m
        //
        // ---------------------------------------------------------------------------

        // np = offset of right of B and V
        np = min(n - l, n);
        // kp = offset of right of work and bottom of V
        kp = min(l, k);

        B_1 = dB;
        B_2 = dB(0, np);

        V_11 = dV;
        V_21 = dV(kp, 0);
        V_12 = dV(0, np);
        V_22 = dV(kp, np);
        V_1x = V_11;
        V_2x = V_21;
        V_x1 = V_11;
        V_x2 = V_12;

        work_1 = work;
        work_2 = work(0, kp);

        // work = B V^H
        // work_1 = B_2
        blas::device_memcpy_2d( work_1, ldwork, B_2, lddb, m, l, queue );
        // work_1 = work_1 V_12^H = B_2 V_12^H
        blas::trmm( Layout::ColMajor, Side::Right, Uplo::Lower, op_trans, Diag::NonUnit, m, l, one, V_12, lddv, work_1, ldwork, queue );
        // work_1 = B_1 V_11^H + work_1
        blas::gemm( Layout::ColMajor, Op::NoTrans, op_trans, m, l, n - l, one, B_1, lddb, V_11, lddv, one, work_1, ldwork, queue );
        // work_2 = B V_2x^H + work_2
        blas::gemm( Layout::ColMajor, Op::NoTrans, op_trans, m, k - l, n, one, dB, lddb, V_2x, lddv, zero, work_2, ldwork, queue );

        // work = A + B V^H
        // work = work + A
        blas::geadd( Layout::ColMajor, Op::NoTrans, m, k, one, dA, ldda, one, work, ldwork, queue );

        // work = (A + B V^H) op(T)
        // work = work op(T)
        blas::trmm( Layout::ColMajor, Side::Right, Uplo::Upper, trans, Diag::NonUnit, m, k, one, dT, lddt, work, ldwork, queue );

        // A = A - (A + B V^H) op(T)
        // A = A - work
        blas::geadd( Layout::ColMajor, Op::NoTrans, m, k, -one, work, ldwork, one, dA, ldda, queue );

        // B = B - (A + B V^H) op(T) V = B - work V
        // B_1 = -work V_x1 + B_1
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n - l, k, -one, work, ldwork, V_x1, lddv, one, B_1, lddb, queue );
        // B_2 = -work_2 V_22 + B_2
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, l, k - l, -one, work_2, ldwork, V_22, lddv, one, B_2, lddb, queue );
        // work_1 = work_1 V_12
        blas::trmm( Layout::ColMajor, Side::Right, Uplo::Lower, Op::NoTrans, Diag::NonUnit, m, l, one, V_12, lddv, work_1, ldwork, queue );
        // B_2 = B_2 - work_1
        blas::geadd( Layout::ColMajor, Op::NoTrans, m, l, -one, work_1, ldwork, one, B_2, lddb, queue );

        // ---------------------------------------------------------------------------
    }
    else if (row && backward && left) {

        // ---------------------------------------------------------------------------
        //
        //        Let  W =  [ V I ] ( I is k-by-k, V is k-by-m )
        //
        //        Form  op(H) C  where  C = [ B ]  (m-by-n)
        //                                  [ A ]  (k-by-n)
        //
        //        H = I - W^H op(T) W
        //
        //        A = A -     op(T) (A + V B)
        //        B = B - V^H op(T) (A + V B)
        //
        // ---------------------------------------------------------------------------
        //
        //                 n
        //        B    = [ B_1 ] l
        //               [ B_2 ] m-l
        //
        //                 l     m-l
        //        V    = [ V_11  V_12 ] k-l
        //               [ V_21  V_22 ] l
        //
        //                 n
        //        work = [ work_1 ] k-l
        //               [ work_2 ] l
        //
        // ---------------------------------------------------------------------------

        // mp = offset of bottom of B and right of V
        mp = min(l, m);
        // kp = offset of bottom of work and V
        kp = min(k - l, k);

        B_1 = dB;
        B_2 = dB(mp, 0);

        V_11 = dV;
        V_21 = dV(kp, 0);
        V_12 = dV(0, mp);
        V_22 = dV(kp, mp);
        V_1x = V_11;
        V_2x = V_21;
        V_x1 = V_11;
        V_x2 = V_12;

        work_1 = work;
        work_2 = work(kp, 0);

        // work = V B
        // work_2 = B_1
        blas::device_memcpy_2d( work_2, ldwork, B_1, lddb, l, n, queue );
        // work_2 = V_21 work_2  V_21 B_1
        blas::trmm( Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, l, n, one, V_21, lddv, work_2, ldwork, queue );
        // work_2 = V_22 B_2 + work_2
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, l, n, m - l, one, V_22, lddv, B_1, lddb, one, work_2, ldwork, queue );
        // work_1 = V_1x B + work_1
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, k - l, n, m, one, V_1x, lddv, dB, lddb, zero, work_1, ldwork, queue );

        // work = A + V B
        // work = work + A
        blas::geadd( Layout::ColMajor, Op::NoTrans, k, n, one, dA, ldda, one, work, ldwork, queue );

        // work = op(T) (A + V B)
        // work = op(T) work
        blas::trmm( Layout::ColMajor, Side::Left, Uplo::Lower, trans, Diag::NonUnit, k, n, one, dT, lddt, work, ldwork, queue );

        // A = A - op(T) (A + V B)
        // A = A - work
        blas::geadd( Layout::ColMajor, Op::NoTrans, k, n, -one, work, ldwork, one, dA, ldda, queue );

        // B = B - V^H op(T) (A + V B) = B - V^H work
        // B_2 = V_x2^H work + B_2
        blas::gemm( Layout::ColMajor, op_trans, Op::NoTrans, m - l, n, k, -one, V_x2, lddv, work, ldwork, one, B_2, lddb, queue );
        // B_1 = -V_11^H work_1 + B_1
        blas::gemm( Layout::ColMajor, op_trans, Op::NoTrans, l, n, k - l, -one, V_11, lddv, work_1, ldwork, one, B_1, lddb, queue );
        // work_2 = V_21^H work_2
        blas::trmm( Layout::ColMajor, Side::Left, Uplo::Upper, op_trans, Diag::NonUnit, l, n, one, V_21, lddv, work_2, ldwork, queue );
        // B_1 = B_1 - work_2
        blas::geadd( Layout::ColMajor, Op::NoTrans, l, n, -one, work_2, ldwork, one, B_1, lddb, queue );

        // ---------------------------------------------------------------------------
    }
    else if (row && backward && right) {

        // ---------------------------------------------------------------------------
        //
        //        Let  W =  [ V I ] ( I is k-by-k, V is k-by-n )
        //
        //        Form  C op(H)  where  C = [ B A ] (A is m-by-k, B is m-by-n)
        //
        //        H = I - W^H op(T) W
        //
        //        A = A - (A + B V^H) op(T)
        //        B = B - (A + B V^H) op(T) V
        //
        // ---------------------------------------------------------------------------
        //
        //                 l      n-l
        //        B    = [ B_1    B_2 ] m
        //
        //                 l      n-l
        //        V    = [ V_11   V_12 ] k-l
        //               [ V_21   V_22 ] l
        //
        //                 k-l    l
        //        work = [ work_1 work2 ] m
        //
        // ---------------------------------------------------------------------------

        // np = offset of right of B and V
        np = min(l, n);
        // kp = offset of right of work and bottom of V
        kp = min(k - l, k);

        B_1 = dB;
        B_2 = dB(0, np);

        V_11 = dV;
        V_21 = dV(kp, 0);
        V_12 = dV(0, np);
        V_22 = dV(kp, np);
        V_1x = V_11;
        V_2x = V_21;
        V_x1 = V_11;
        V_x2 = V_12;

        work_1 = work;
        work_2 = work(0, kp);

        // work = B V^H
        // work_2 = B_1
        blas::device_memcpy_2d( work_2, ldwork, B_1, lddb, m, l, queue );
        // work_2 = work_2 V_21^H = B_1 V_21^H
        blas::trmm( Layout::ColMajor, Side::Right, Uplo::Upper, op_trans, Diag::NonUnit, m, l, one, V_21, lddv, work_2, ldwork, queue );
        // work_2 = B_2 V_22^H + work_2
        blas::gemm( Layout::ColMajor, Op::NoTrans, op_trans, m, l, n - l, one, B_1, lddb, V_22, lddv, one, work_2, ldwork, queue );
        // work_1 = B V_1x^H + work_1
        blas::gemm( Layout::ColMajor, Op::NoTrans, op_trans, m, k - l, n, one, dB, lddb, V_1x, lddv, zero, work_1, ldwork, queue );

        // work = A + B V^H
        // work = A + work
        blas::geadd( Layout::ColMajor, Op::NoTrans, m, k, one, dA, ldda, one, work, ldwork, queue );

        // work = (A + B V^H) op(T)
        // work = work op(T)
        blas::trmm( Layout::ColMajor, Side::Right, Uplo::Lower, trans, Diag::NonUnit, m, k, one, dT, lddt, work, ldwork, queue );

        // A = A - (A + B V^H) op(T)
        // A = A - work
        blas::geadd( Layout::ColMajor, Op::NoTrans, m, k, -one, work, ldwork, one, dA, ldda, queue );

        // B = B - (A + B V^H) op(T) V = B - work V
        // B_2 = -work V_x2 + B_2
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n - l, k, -one, work, ldwork, V_x2, lddv, one, B_2, lddb, queue );
        // B_1 = -work_1 V_11 + B_1
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, l, k - l, -one, work_1, ldwork, V_11, lddv, one, B_1, lddb, queue );
        // work_2 = work_2 V_21
        blas::trmm( Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, l, one, V_21, lddv, work_2, ldwork, queue );
        // B_1 = B_1 - work_2
        blas::geadd( Layout::ColMajor, Op::NoTrans, m, l, -one, work_2, ldwork, one, B_1, lddb, queue );
    }

    queue.sync();
    blas::device_free( work, queue );

    return;
}

template void tprfb(
    lapack::Side side, lapack::Op trans,
    lapack::Direction direction, lapack::StoreV storev,
    int64_t m, int64_t n, int64_t k, int64_t l,
    float const* dV, int64_t lddv,
    float const* dT, int64_t lddt,
    float* dA, int64_t ldda,
    float* dB, int64_t lddb,
    lapack::Queue& queue );

template void tprfb(
    lapack::Side side, lapack::Op trans,
    lapack::Direction direction, lapack::StoreV storev,
    int64_t m, int64_t n, int64_t k, int64_t l,
    double const* dV, int64_t lddv,
    double const* dT, int64_t lddt,
    double* dA, int64_t ldda,
    double* dB, int64_t lddb,
    lapack::Queue& queue );

template void tprfb(
    lapack::Side side, lapack::Op trans,
    lapack::Direction direction, lapack::StoreV storev,
    int64_t m, int64_t n, int64_t k, int64_t l,
    std::complex<float> const* dV, int64_t lddv,
    std::complex<float> const* dT, int64_t lddt,
    std::complex<float>* dA, int64_t ldda,
    std::complex<float>* dB, int64_t lddb,
    lapack::Queue& queue );

template void tprfb(
    lapack::Side side, lapack::Op trans,
    lapack::Direction direction, lapack::StoreV storev,
    int64_t m, int64_t n, int64_t k, int64_t l,
    std::complex<double> const* dV, int64_t lddv,
    std::complex<double> const* dT, int64_t lddt,
    std::complex<double>* dA, int64_t ldda,
    std::complex<double>* dB, int64_t lddb,
    lapack::Queue& queue );

} // namespace lapack
