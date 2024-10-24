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
/// @ingroup geev_computational
int64_t hseqr(
    lapack::JobSchur jobschur, lapack::Job compz, int64_t n, int64_t ilo, int64_t ihi,
    float* H, int64_t ldh,
    std::complex<float>* W,
    float* Z, int64_t ldz )
{
    char jobschur_ = to_char( jobschur );
    char compz_ = to_char_comp( compz );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ilo_ = to_lapack_int( ilo );
    lapack_int ihi_ = to_lapack_int( ihi );
    lapack_int ldh_ = to_lapack_int( ldh );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    // split-complex representation
    lapack::vector< float > WR( max( 1, n ) );
    lapack::vector< float > WI( max( 1, n ) );

    // query for workspace size
    float qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_shseqr(
        &jobschur_, &compz_, &n_, &ilo_, &ihi_,
        H, &ldh_,
        &WR[0],
        &WI[0],
        Z, &ldz_,
        qry_work, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< float > work( lwork_ );

    LAPACK_shseqr(
        &jobschur_, &compz_, &n_, &ilo_, &ihi_,
        H, &ldh_,
        &WR[0],
        &WI[0],
        Z, &ldz_,
        &work[0], &lwork_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    // merge split-complex representation
    for (int64_t i = 0; i < n; ++i) {
        W[i] = std::complex<float>( WR[i], WI[i] );
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t hseqr(
    lapack::JobSchur jobschur, lapack::Job compz, int64_t n, int64_t ilo, int64_t ihi,
    double* H, int64_t ldh,
    std::complex<double>* W,
    double* Z, int64_t ldz )
{
    char jobschur_ = to_char( jobschur );
    char compz_ = to_char_comp( compz );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ilo_ = to_lapack_int( ilo );
    lapack_int ihi_ = to_lapack_int( ihi );
    lapack_int ldh_ = to_lapack_int( ldh );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    // split-complex representation
    lapack::vector< double > WR( max( 1, n ) );
    lapack::vector< double > WI( max( 1, n ) );

    // query for workspace size
    double qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_dhseqr(
        &jobschur_, &compz_, &n_, &ilo_, &ihi_,
        H, &ldh_,
        &WR[0],
        &WI[0],
        Z, &ldz_,
        qry_work, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< double > work( lwork_ );

    LAPACK_dhseqr(
        &jobschur_, &compz_, &n_, &ilo_, &ihi_,
        H, &ldh_,
        &WR[0],
        &WI[0],
        Z, &ldz_,
        &work[0], &lwork_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    // merge split-complex representation
    for (int64_t i = 0; i < n; ++i) {
        W[i] = std::complex<float>( WR[i], WI[i] );
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup geev_computational
int64_t hseqr(
    lapack::JobSchur jobschur, lapack::Job compz, int64_t n, int64_t ilo, int64_t ihi,
    std::complex<float>* H, int64_t ldh,
    std::complex<float>* W,
    std::complex<float>* Z, int64_t ldz )
{
    char jobschur_ = to_char( jobschur );
    char compz_ = to_char_comp( compz );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ilo_ = to_lapack_int( ilo );
    lapack_int ihi_ = to_lapack_int( ihi );
    lapack_int ldh_ = to_lapack_int( ldh );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_chseqr(
        &jobschur_, &compz_, &n_, &ilo_, &ihi_,
        (lapack_complex_float*) H, &ldh_,
        (lapack_complex_float*) W,
        (lapack_complex_float*) Z, &ldz_,
        (lapack_complex_float*) qry_work, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<float> > work( lwork_ );

    LAPACK_chseqr(
        &jobschur_, &compz_, &n_, &ilo_, &ihi_,
        (lapack_complex_float*) H, &ldh_,
        (lapack_complex_float*) W,
        (lapack_complex_float*) Z, &ldz_,
        (lapack_complex_float*) &work[0], &lwork_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the eigenvalues of a Hessenberg matrix H
/// and, optionally, the matrices T and Z from the Schur decomposition
/// \[
///     H = Z T Z^H,
/// \]
/// where T is an upper triangular matrix (the
/// Schur form), and Z is the unitary matrix of Schur vectors.
///
/// Optionally Z may be postmultiplied into an input unitary
/// matrix Q so that this routine can give the Schur factorization
/// of a matrix A which has been reduced to the Hessenberg form H
/// by the unitary matrix Q: $A = Q H Q^H = (QZ) T (QZ)^H$.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] jobschur
///     - lapack::JobSchur::None:  compute eigenvalues only;
///     - lapack::JobSchur::Schur: compute eigenvalues and the Schur form T.
///
/// @param[in] compz
///     - lapack::CompQ::NoVec:
///         no Schur vectors are computed;
///     - lapack::CompQ::Vec:
///         Z is initialized to the unit matrix and the matrix Z
///         of Schur vectors of H is returned;
///     - lapack::CompQ::Update:
///         Z must contain an unitary matrix Q on entry, and
///         the product $Q Z$ is returned.
///
/// @param[in] n
///     The order of the matrix H. n >= 0.
///
/// @param[in] ilo
///
/// @param[in] ihi
///     It is assumed that H is already upper triangular in rows
///     and columns 1:ilo-1 and ihi+1:n. ilo and ihi are normally
///     set by a previous call to `lapack::gebal`, and then passed to `lapack::gehrd`
///     when the matrix output by `lapack::gebal` is reduced to Hessenberg
///     form. Otherwise ilo and ihi should be set to 1 and n
///     respectively.
///     - If n > 0, then 1 <= ilo <= ihi <= n;
///     - if n = 0, then ilo = 1 and ihi = 0.
///
/// @param[in,out] H
///     The n-by-n matrix H, stored in an ldh-by-n array.
///     On entry, the upper Hessenberg matrix H.
///     On exit, if successful and job = Schur, H contains the upper
///     triangular matrix T from the Schur decomposition (the
///     Schur form). If successful and job = None, the contents of
///     H are unspecified on exit. (The output value of H when
///     return value > 0 is given under the description of info below.)
///     \n
///     Unlike earlier versions of `hseqr`, this subroutine may
///     explicitly H(i,j) = 0 for i > j and j = 1, 2, ... ilo-1
///     or j = ihi+1, ihi+2, ... n.
///
/// @param[in] ldh
///     The leading dimension of the array H. ldh >= max(1,n).
///
/// @param[out] W
///     The vector W of length n.
///     The computed eigenvalues. If job = Schur, the eigenvalues are
///     stored in the same order as on the diagonal of the Schur
///     form returned in H, with W(i) = H(i,i).
///     \n
///     Note: In LAPACK++, W is always complex, whereas LAPACK with a
///     real matrix H uses a split-complex representation (WR, WI) for W.
///
/// @param[in,out] Z
///     The n-by-n matrix Z, stored in an ldz-by-n array.
///     - If compz = NoVec, Z is not referenced.
///
///     - If compz = Vec, on entry Z need not be set and on exit,
///     if successful, Z contains the unitary matrix Z of the Schur
///     vectors of H.
///
///     - If compz = Update, on entry Z must contain an
///     n-by-n matrix Q, which is assumed to be equal to the unit
///     matrix except for the submatrix Z(ilo:ihi,ilo:ihi). On exit,
///     if successful, Z contains $Q Z$.
///     Normally Q is the unitary matrix generated by `lapack::unghr`
///     after the call to `lapack::gehrd` which formed the Hessenberg matrix H.
///
///     - The output value of Z when return value > 0 is given under
///     the description of info below.
///
/// @param[in] ldz
///     The leading dimension of the array Z. if compz = Vec or
///     compz = Update, then ldz >= max(1,n). Otherwize, ldz >= 1.
///
/// @return = 0: successful exit
/// @return > 0: if return value = i, `hseqr` failed to compute all of
///     the eigenvalues. Elements 1:ilo-1 and i+1:n of W
///     contain those eigenvalues which have been
///     successfully computed. (Failures are rare.)
///     \n
///     If return value > 0 and job = None, then on exit, the
///     remaining unconverged eigenvalues are the eigen-
///     values of the upper Hessenberg matrix rows and
///     columns ilo through info of the final, output
///     value of H.
///     \n
///     If return value > 0 and job = Schur, then on exit
///     \n
///     (*) (initial value of H)*$U = U$ (final value of H)
///     \n
///     where U is a unitary matrix. The final
///     value of H is upper Hessenberg and triangular in
///     rows and columns info+1 through ihi.
///     \n
///     If return value > 0 and compz = Update, then on exit
///     \n
///     (final value of Z) = (initial value of Z)*U
///     \n
///     where U is the unitary matrix in (*) (regard-
///     less of the value of JOBSCHUR.)
///     \n
///     If return value > 0 and compz = Vec, then on exit
///     (final value of Z) = U
///     where U is the unitary matrix in (*) (regard-
///     less of the value of JOBSCHUR.)
///     \n
///     If return value > 0 and compz = NoVec, then Z is not
///     accessed.
///
/// @ingroup geev_computational
int64_t hseqr(
    lapack::JobSchur jobschur, lapack::Job compz, int64_t n, int64_t ilo, int64_t ihi,
    std::complex<double>* H, int64_t ldh,
    std::complex<double>* W,
    std::complex<double>* Z, int64_t ldz )
{
    char jobschur_ = to_char( jobschur );
    char compz_ = to_char_comp( compz );
    lapack_int n_ = to_lapack_int( n );
    lapack_int ilo_ = to_lapack_int( ilo );
    lapack_int ihi_ = to_lapack_int( ihi );
    lapack_int ldh_ = to_lapack_int( ldh );
    lapack_int ldz_ = to_lapack_int( ldz );
    lapack_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    lapack_int ineg_one = -1;
    LAPACK_zhseqr(
        &jobschur_, &compz_, &n_, &ilo_, &ihi_,
        (lapack_complex_double*) H, &ldh_,
        (lapack_complex_double*) W,
        (lapack_complex_double*) Z, &ldz_,
        (lapack_complex_double*) qry_work, &ineg_one, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    lapack::vector< std::complex<double> > work( lwork_ );

    LAPACK_zhseqr(
        &jobschur_, &compz_, &n_, &ilo_, &ihi_,
        (lapack_complex_double*) H, &ldh_,
        (lapack_complex_double*) W,
        (lapack_complex_double*) Z, &ldz_,
        (lapack_complex_double*) &work[0], &lwork_, &info_
    );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
