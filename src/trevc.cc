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
int64_t trevc(
    lapack::Sides sides, lapack::HowMany howmany,
    bool* select, int64_t n,
    float const* T, int64_t ldt,
    float* VL, int64_t ldvl,
    float* VR, int64_t ldvr, int64_t mm,
    int64_t* m )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvl) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvr) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(mm) > std::numeric_limits<lapack_int>::max() );
    }
    char sides_ = sides2char( sides );
    char howmany_ = howmany2char( howmany );

    // lapack_logical (32 or 64-bit) copy
    std::vector< lapack_logical > select_( &select[0], &select[(n)] );
    lapack_logical* select_ptr = &select_[0];

    lapack_int n_ = (lapack_int) n;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldvl_ = (lapack_int) ldvl;
    lapack_int ldvr_ = (lapack_int) ldvr;
    lapack_int mm_ = (lapack_int) mm;
    lapack_int m_ = (lapack_int) *m;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (3*n) );

    LAPACK_strevc(
        &sides_, &howmany_,
        select_ptr, &n_,
        T, &ldt_,
        VL, &ldvl_,
        VR, &ldvr_, &mm_, &m_,
        &work[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    // [sd]trevc update select
    std::copy( select_.begin(), select_.end(), select );
    *m = m_;
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t trevc(
    lapack::Sides sides, lapack::HowMany howmany,
    bool* select, int64_t n,
    double const* T, int64_t ldt,
    double* VL, int64_t ldvl,
    double* VR, int64_t ldvr, int64_t mm,
    int64_t* m )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvl) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvr) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(mm) > std::numeric_limits<lapack_int>::max() );
    }
    char sides_ = sides2char( sides );
    char howmany_ = howmany2char( howmany );

    // lapack_logical (32 or 64-bit) copy
    std::vector< lapack_logical > select_( &select[0], &select[(n)] );
    lapack_logical* select_ptr = &select_[0];

    lapack_int n_ = (lapack_int) n;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldvl_ = (lapack_int) ldvl;
    lapack_int ldvr_ = (lapack_int) ldvr;
    lapack_int mm_ = (lapack_int) mm;
    lapack_int m_ = (lapack_int) *m;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (3*n) );

    LAPACK_dtrevc(
        &sides_, &howmany_,
        select_ptr, &n_,
        T, &ldt_,
        VL, &ldvl_,
        VR, &ldvr_, &mm_, &m_,
        &work[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    // [sd]trevc update select
    std::copy( select_.begin(), select_.end(), select );
    *m = m_;
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t trevc(
    lapack::Sides sides, lapack::HowMany howmany,
    bool const* select, int64_t n,
    std::complex<float>* T, int64_t ldt,
    std::complex<float>* VL, int64_t ldvl,
    std::complex<float>* VR, int64_t ldvr, int64_t mm,
    int64_t* m )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvl) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvr) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(mm) > std::numeric_limits<lapack_int>::max() );
    }
    char sides_ = sides2char( sides );
    char howmany_ = howmany2char( howmany );

    // lapack_logical (32 or 64-bit) copy
    std::vector< lapack_logical > select_( &select[0], &select[(n)] );
    lapack_logical const* select_ptr = &select_[0];

    lapack_int n_ = (lapack_int) n;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldvl_ = (lapack_int) ldvl;
    lapack_int ldvr_ = (lapack_int) ldvr;
    lapack_int mm_ = (lapack_int) mm;
    lapack_int m_ = (lapack_int) *m;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (2*n) );
    lapack::vector< float > rwork( (n) );

    LAPACK_ctrevc(
        &sides_, &howmany_,
        select_ptr, &n_,
        (lapack_complex_float*) T, &ldt_,
        (lapack_complex_float*) VL, &ldvl_,
        (lapack_complex_float*) VR, &ldvr_, &mm_, &m_,
        (lapack_complex_float*) &work[0],
        &rwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes some or all of the right and/or left eigenvectors of
/// a complex upper triangular matrix T.
/// Matrices of this type are produced by the Schur factorization of
/// a complex general matrix: $A = Q T Q^H$, as computed by `lapack::hseqr`.
///
/// The right eigenvector x and the left eigenvector y of T corresponding
/// to an eigenvalue $\lambda$ are defined by:
/// \[
///     T x = \lambda x,
/// \]
/// \[
///     y^H T = \lambda y^H,
/// \]
/// where $y^H$ denotes the conjugate transpose of the vector y.
/// The eigenvalues are not input to this routine, but are read directly
/// from the diagonal of T.
///
/// This routine returns the matrices X and/or Y of right and left
/// eigenvectors of T, or the products $Q X$ and/or $Q Y$, where Q is an
/// input matrix. If Q is the unitary factor that reduces a matrix A to
/// Schur form T, then $Q X$ and $Q Y$ are the matrices of right and left
/// eigenvectors of A.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] sides
///     - lapack::Sides::Right: compute right eigenvectors only;
///     - lapack::Sides::Left:  compute left eigenvectors only;
///     - lapack::Sides::Both:  compute both right and left eigenvectors.
///
/// @param[in] howmany
///     - lapack::HowMany::All:
///         compute all right and/or left eigenvectors;
///     - lapack::HowMany::Backtransform:
///         compute all right and/or left eigenvectors,
///         backtransformed using the matrices supplied in
///         VR and/or VL;
///     - lapack::HowMany::Select:
///         compute selected right and/or left eigenvectors,
///         as indicated by the logical array select.
///
/// @param[in] select
///     The vector select of length n.
///     If howmany = Select, select specifies the eigenvectors to be
///     computed.
///     The eigenvector corresponding to the j-th eigenvalue is
///     computed if select(j) = true.
///     Not referenced if howmany = All or Backtransform.
///     TODO: updated in real case. See [sd]trevc.
///
/// @param[in] n
///     The order of the matrix T. n >= 0.
///
/// @param[in,out] T
///     The n-by-n matrix T, stored in an ldt-by-n array.
///     The upper triangular matrix T. T is modified, but restored
///     on exit.
///
/// @param[in] ldt
///     The leading dimension of the array T. ldt >= max(1,n).
///
/// @param[in,out] VL
///     The n-by-mm matrix VL, stored in an ldvl-by-mm array.
///     On entry, if side = Left or Both and howmany = Backtransform, VL must
///     contain an n-by-n matrix Q (usually the unitary matrix Q of
///     Schur vectors returned by `lapack::hseqr`).
///     On exit, if side = Left or Both, VL contains:
///     - if howmany = All, the matrix Y of left eigenvectors of T;
///     - if howmany = Backtransform, the matrix $Q Y$;
///     - if howmany = Select, the left eigenvectors of T specified by
///         select, stored consecutively in the columns
///         of VL, in the same order as their
///         eigenvalues.
///     \n
///     Not referenced if side = Right.
///
/// @param[in] ldvl
///     The leading dimension of the array VL.
///     ldvl >= 1, and if side = Left or Both, ldvl >= n.
///
/// @param[in,out] VR
///     The n-by-mm matrix VR, stored in an ldvr-by-mm array.
///     On entry, if side = Right or Both and howmany = Backtransform, VR must
///     contain an n-by-n matrix Q (usually the unitary matrix Q of
///     Schur vectors returned by `lapack::hseqr`).
///     On exit, if side = Right or Both, VR contains:
///     - if howmany = All, the matrix X of right eigenvectors of T;
///     - if howmany = Backtransform, the matrix $Q X$;
///     - if howmany = Select, the right eigenvectors of T specified by
///         select, stored consecutively in the columns
///         of VR, in the same order as their
///         eigenvalues.
///     \n
///     Not referenced if side = Left.
///
/// @param[in] ldvr
///     The leading dimension of the array VR.
///     ldvr >= 1, and if side = Right or Both, ldvr >= n.
///
/// @param[in] mm
///     The number of columns in the arrays VL and/or VR. mm >= m.
///
/// @param[out] m
///     The number of columns in the arrays VL and/or VR actually
///     used to store the eigenvectors.
///     If howmany = All or Backtransform, m is set to n.
///     Each selected eigenvector occupies one column.
///
/// @return = 0: successful exit
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// The algorithm used in this program is basically backward (forward)
/// substitution, with scaling to make the the code robust against
/// possible overflow.
///
/// Each eigenvector is normalized so that the element of largest
/// magnitude has magnitude 1; here the magnitude of a complex number
/// (x,y) is taken to be |x| + |y|.
///
/// @ingroup geev_computational
int64_t trevc(
    lapack::Sides sides, lapack::HowMany howmany,
    bool const* select, int64_t n,
    std::complex<double>* T, int64_t ldt,
    std::complex<double>* VL, int64_t ldvl,
    std::complex<double>* VR, int64_t ldvr, int64_t mm,
    int64_t* m )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvl) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldvr) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(mm) > std::numeric_limits<lapack_int>::max() );
    }
    char sides_ = sides2char( sides );
    char howmany_ = howmany2char( howmany );

    // lapack_logical (32 or 64-bit) copy
    std::vector< lapack_logical > select_( &select[0], &select[(n)] );
    lapack_logical const* select_ptr = &select_[0];

    lapack_int n_ = (lapack_int) n;
    lapack_int ldt_ = (lapack_int) ldt;
    lapack_int ldvl_ = (lapack_int) ldvl;
    lapack_int ldvr_ = (lapack_int) ldvr;
    lapack_int mm_ = (lapack_int) mm;
    lapack_int m_ = (lapack_int) *m;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (2*n) );
    lapack::vector< double > rwork( (n) );

    LAPACK_ztrevc(
        &sides_, &howmany_,
        select_ptr, &n_,
        (lapack_complex_double*) T, &ldt_,
        (lapack_complex_double*) VL, &ldvl_,
        (lapack_complex_double*) VR, &ldvr_, &mm_, &m_,
        (lapack_complex_double*) &work[0],
        &rwork[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1
        #endif
    );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    return info_;
}

}  // namespace lapack
