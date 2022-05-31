// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t unmhr(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t ilo, int64_t ihi,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* tau,
    std::complex<float>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ilo) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ihi) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int ilo_ = (lapack_int) ilo;
    lapack_int ihi_ = (lapack_int) ihi;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldc_ = (lapack_int) ldc;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_cunmhr(
        &side_, &trans_, &m_, &n_, &ilo_, &ihi_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) C, &ldc_,
        (lapack_complex_float*) qry_work, &ineg_one, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );

    LAPACK_cunmhr(
        &side_, &trans_, &m_, &n_, &ilo_, &ihi_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) C, &ldc_,
        (lapack_complex_float*) &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Multiplies the general m-by-n matrix C by Q from `lapack::gehrd` as follows:
///
/// - side = Left,  trans = NoTrans:   $Q C$
/// - side = Right, trans = NoTrans:   $C Q$
/// - side = Left,  trans = ConjTrans: $Q^H C$
/// - side = Right, trans = ConjTrans: $C Q^H$
///
/// where Q is a unitary matrix of order m if
/// side = Left and order n if side = Right. Q is defined as the product of
/// ihi-ilo elementary reflectors, as returned by `lapack::gehrd`:
/// \[
///     Q = H(ilo) H(ilo+1) \dots H(ihi-1).
/// \]
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this is an alias for `lapack::ormhr`.
///
/// @param[in] side
///     - lapack::Side::Left:  apply $Q$ or $Q^H$ from the Left;
///     - lapack::Side::Right: apply $Q$ or $Q^H$ from the Right.
///
/// @param[in] trans
///     - lapack::Op::NoTrans:   apply $Q$   (No transpose)
///     - lapack::Op::ConjTrans: apply $Q^H$ (Conjugate transpose)
///
/// @param[in] m
///     The number of rows of the matrix C. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix C. n >= 0.
///
/// @param[in] ilo
///
/// @param[in] ihi
///     ilo and ihi must have the same values as in the previous call
///     of `lapack::gehrd`. Q is equal to the unit matrix except in the
///     submatrix Q(ilo+1:ihi,ilo+1:ihi).
///     - If side = Left  and m > 0, then 1 <= ilo <= ihi <= m;
///     - if side = Left  and m = 0, then ilo = 1 and ihi = 0;
///     - if side = Right and n > 0, then 1 <= ilo <= ihi <= n;
///     - if side = Right and n = 0, then ilo = 1 and ihi = 0.
///
/// @param[in] A
///     The vectors which define the elementary reflectors, as
///     returned by `lapack::gehrd`.
///     - If side = Left,  the m-by-m matrix A, stored in an lda-by-m array;
///     - if side = Right, the n-by-n matrix A, stored in an lda-by-n array.
///
/// @param[in] lda
///     The leading dimension of the array A.
///     - If side = Left,  lda >= max(1,m);
///     - If side = Right, lda >= max(1,n).
///
/// @param[in] tau
///     tau(i) must contain the scalar factor of the elementary
///     reflector H(i), as returned by `lapack::gehrd`.
///     - If side = Left,  the vector tau of length m-1;
///     - if side = Right, the vector tau of length n-1.
///
/// @param[in,out] C
///     The m-by-n matrix C, stored in an ldc-by-n array.
///     On entry, the m-by-n matrix C.
///     On exit, C is overwritten by
///     $Q C$ or $Q^H C$ or $C Q^H$ or $C Q$.
///
/// @param[in] ldc
///     The leading dimension of the array C. ldc >= max(1,m).
///
/// @return = 0: successful exit
///
/// @ingroup geev_computational
int64_t unmhr(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t ilo, int64_t ihi,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* tau,
    std::complex<double>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ilo) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ihi) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int ilo_ = (lapack_int) ilo;
    lapack_int ihi_ = (lapack_int) ihi;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldc_ = (lapack_int) ldc;
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_zunmhr(
        &side_, &trans_, &m_, &n_, &ilo_, &ihi_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) C, &ldc_,
        (lapack_complex_double*) qry_work, &ineg_one, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );

    LAPACK_zunmhr(
        &side_, &trans_, &m_, &n_, &ilo_, &ihi_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) C, &ldc_,
        (lapack_complex_double*) &work[0], &lwork_, &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
