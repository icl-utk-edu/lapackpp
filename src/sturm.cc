// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {
//------------------------------------------------------------------------------
/// @ingroup heev_computational
/// sturm computes a Scaled Sturm Sequence using a real symmetric tri-diagonal
/// matrix. NOTE this calls no LAPACK routine; the code is here. Only single
/// and double precision code exist.
///
/// Parameters: All parameters are read-only by this routine.
///  @param[in]  diag: a vector of 'n' diagonal elements.
///  @param[in]  offd: a vector of 'n-1' off-diagonal elements.
///  @param[in]     n: The order of the matrix.
///  @param[in]     u: The sigma test point.
///  @retval     Number of eigenvalues strictly less than u.
///
// This code is adapted from the PLASMA library.
//
// See https://archive.siam.org/meetings/la03/proceedings/zhangjy3.pdf
// "J. Zhang, 2003, The Scaled Sturm Sequence Computation".  Both the Sturm
// (proportional) and the classical Sturm can suffer from underflow and
// overflow for some problematic matrices; automatic rescaling avoids that.
// using the classical Sturm as the starting point avoids division (and
// checking to avoid division by zero). Computation is still O(N) but about
// 1.5 times more flops.
//
// diag[0..n-1] are the diagonals; offd[0..n-2] are the offdiagonals.
// The classic recurrence: (u is the \lambda cutpoint in question).
// we never index p by less than 0, p[-1] is a dummy handled outside the loop.
// Note this is zero-relative indexing.
// p[-1] = 1.;
// p[0] = diag[0] - u;
// p[i] = (diag[i]-u)*p[i-1] - offd[i-1]*offd[i-1]*p[i-2], i=1, N-1.
//
// The Classical Sturm recurrence can be shown as a matrix computation; namely
// P[i] = M[i]*P[i-1].
// M[i] = [(diag[i]-u) , -offd[i-1]*offd[i-1] ] and P[i-1] = [ p[i-1] ]
//        [          1 ,                    0 ]              [ p[i-2] ]
// with P[-1] defined to be [1, 0] transposed.
// notice 'p' is elements of the classical Sturm sequence, 'P' is a vector.
//
// the matrix computation results in the vector:
// M[i]*P[i-1] = { (diag[i]-u)*p[i-1] -offd[i-1]*offd[i-1]*p[i-2] , p[i-1] }
//
// So, in the classical case, P[i][0] is the classic Sturm sequence for p[i];
// the second element is just the classic Sturm for p[i-1].
//
// However, this won't remain that way. For the SCALED Sturm sequence, we
// will scale P[i] after each calculation, with the scalar 's':
//
//----------------------------------
// P[i] = s * M[i]*P[i-1], i=0, N-1. Note we are scaling a vector here.
//----------------------------------
//
// For code, we represent P[i-1] as two scalars, [Pm1_0 , Pm1_1].
// The matrix calculation is thus:
// M[i]*P[i-1] = { (diag[i]-u)*Pm1_0 -offd[i-1]*offd[i-1]*Pm1_1 , Pm1_0 }
// or in three equations, adding in the scalar:
// save = s * Pm1_0;
// Pm1_0 = s * ( (diag[i]-u)*Pm1_0 -offd[i-1]*offd[i-1]*Pm1_1 );
// Pm1_1 = save;

// Pm1_0 is used like the classical Sturm sequence; meaning we must calculate
// sign changes.
//
// s is computed given the vector X[] = M[i]*P[i-1] above.
// PHI is set to 10^{10}, UPSILON is set to 10^{-10}. Then:
//    w = max(fabs(X[0]), fabs(X[1])).
//    if w > PHI then s = PHI/w;
//    else if w < UPSILON then s = UPSILON/w;
//    else s=1.0 (or, do not scale X).
//
// This algorithm is backward stable.
// Execution time is 1.5 times classic Sturm.
//
// No sign change counts eigenvalues >= u.
// sign changes count eigenvalues <  u.
// This routine returns the number of sign changes, which is the count of
// eigenvalues strictly less than u.
//
// computation: What we need for each computation:
// M[i], which we compute on the fly from diag[i] and offd[i-1].
// P[i-1], which has two elements, [Pm1_0, Pm1_1]. (Pm1 means P minus 1).
// LAPACK routine DLAEBZ computes a standard Sturm sequence; there is no
// comparable auto-scaling Sturm sequence.
//
// This routine is most similar to LAPACK DLANEG.f, but is not a replacement
// for it. DLANEG.f does not autoscale.
//------------------------------------------------------------------------------

template <typename scalar_t>
int64_t sturm(
    int64_t n, scalar_t const* diag, scalar_t const* offd, scalar_t u )
{
    int64_t i, isneg=0;
    scalar_t s, w, v0, v1, Pm1_0, Pm1_1, phi, upsilon;
    if (n == 0)
        return 0;

    // todo: adjust for float. Use constants like epsilon.
    phi = scalar_t( int64_t( 1 ) << 34 );  // Roughly 1.718e+10.
    const scalar_t one=1.0;
    upsilon = one/phi;

    Pm1_1 = one;
    Pm1_0 = (diag[0]-u);
    // Our first test.
    if (Pm1_0 < 0)
        isneg = 1;
    for (i = 1; i < n; ++i) {
        // first part of scaling, just get w.
        v0 = fabs(Pm1_0);
        v1 = fabs(Pm1_1);
        if (v0 > v1) w = v0;
        else         w = v1;

        // Go ahead and calculate P[i]:
        s = Pm1_0;
        Pm1_0 = (diag[i]-u)*Pm1_0 -((offd[i-1]*offd[i-1])*Pm1_1);
        Pm1_1 = s;

        // Now determine whether to scale these new values.
        if (w > phi) {  // If the largest magnitude > big,
            s = phi/w;  // compute scaling ratio,
            Pm1_0 *= s; // apply it.
            Pm1_1 *= s;
        }
        else {
            if (w < upsilon) {  // if largest magnitude < tiny,
                s = upsilon/w;  // compute scaling ratio,
                Pm1_0 *= s;     // apply it.
                Pm1_1 *= s;
            }
        }

        // Finally, see if the sign changed.
        if ((Pm1_0 < 0 && Pm1_1 >= 0)
            || (Pm1_0 >= 0 && Pm1_1 < 0))
           isneg++;     // count eigenvalues < u.
    }

    return isneg;
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
int64_t sturm<float>(
    int64_t n, float const* diag, float const* offd, float u );

template
int64_t sturm<double>(
    int64_t n, double const* diag, double const* offd, double u );

} // namespace lapack
