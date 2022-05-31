// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack/fortran.h"
#include "NoConstructAllocator.hh"

#ifdef LAPACK_HAVE_XBLAS

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gbsv_computational
int64_t gbrfsx(
    lapack::Op trans, lapack::Equed equed, int64_t n, int64_t kl, int64_t ku, int64_t nrhs,
    float const* AB, int64_t ldab,
    float const* AFB, int64_t ldafb,
    int64_t const* ipiv,
    float* R,
    float* C,
    float const* B, int64_t ldb,
    float* X, int64_t ldx,
    float* rcond,
    float* berr, int64_t n_err_bnds,
    float* err_bnds_norm,
    float* err_bnds_comp, int64_t nparams,
    float* params )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kl) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ku) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldafb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n_err_bnds) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nparams) > std::numeric_limits<lapack_int>::max() );
    }
    char trans_ = op2char( trans );
    char equed_ = equed2char( equed );
    lapack_int n_ = (lapack_int) n;
    lapack_int kl_ = (lapack_int) kl;
    lapack_int ku_ = (lapack_int) ku;
    lapack_int nrhs_ = (lapack_int) nrhs;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int ldafb_ = (lapack_int) ldafb;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int const* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int const* ipiv_ptr = ipiv;
    #endif
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int ldx_ = (lapack_int) ldx;
    lapack_int n_err_bnds_ = (lapack_int) n_err_bnds;
    lapack_int nparams_ = (lapack_int) nparams;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< float > work( (4*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_sgbrfsx(
        &trans_, &equed_, &n_, &kl_, &ku_, &nrhs_,
        AB, &ldab_,
        AFB, &ldafb_,
        ipiv_ptr,
        R,
        C,
        B, &ldb_,
        X, &ldx_, rcond,
        berr, &n_err_bnds_,
        err_bnds_norm,
        err_bnds_comp, &nparams_,
        params,
        &work[0],
        &iwork[0], &info_
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
/// @ingroup gbsv_computational
int64_t gbrfsx(
    lapack::Op trans, lapack::Equed equed, int64_t n, int64_t kl, int64_t ku, int64_t nrhs,
    double const* AB, int64_t ldab,
    double const* AFB, int64_t ldafb,
    int64_t const* ipiv,
    double* R,
    double* C,
    double const* B, int64_t ldb,
    double* X, int64_t ldx,
    double* rcond,
    double* berr, int64_t n_err_bnds,
    double* err_bnds_norm,
    double* err_bnds_comp, int64_t nparams,
    double* params )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kl) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ku) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldafb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n_err_bnds) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nparams) > std::numeric_limits<lapack_int>::max() );
    }
    char trans_ = op2char( trans );
    char equed_ = equed2char( equed );
    lapack_int n_ = (lapack_int) n;
    lapack_int kl_ = (lapack_int) kl;
    lapack_int ku_ = (lapack_int) ku;
    lapack_int nrhs_ = (lapack_int) nrhs;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int ldafb_ = (lapack_int) ldafb;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int const* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int const* ipiv_ptr = ipiv;
    #endif
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int ldx_ = (lapack_int) ldx;
    lapack_int n_err_bnds_ = (lapack_int) n_err_bnds;
    lapack_int nparams_ = (lapack_int) nparams;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< double > work( (4*n) );
    lapack::vector< lapack_int > iwork( (n) );

    LAPACK_dgbrfsx(
        &trans_, &equed_, &n_, &kl_, &ku_, &nrhs_,
        AB, &ldab_,
        AFB, &ldafb_,
        ipiv_ptr,
        R,
        C,
        B, &ldb_,
        X, &ldx_, rcond,
        berr, &n_err_bnds_,
        err_bnds_norm,
        err_bnds_comp, &nparams_,
        params,
        &work[0],
        &iwork[0], &info_
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
/// @ingroup gbsv_computational
int64_t gbrfsx(
    lapack::Op trans, lapack::Equed equed, int64_t n, int64_t kl, int64_t ku, int64_t nrhs,
    std::complex<float> const* AB, int64_t ldab,
    std::complex<float> const* AFB, int64_t ldafb,
    int64_t const* ipiv,
    float* R,
    float* C,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float>* X, int64_t ldx,
    float* rcond,
    float* berr, int64_t n_err_bnds,
    float* err_bnds_norm,
    float* err_bnds_comp, int64_t nparams,
    float* params )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kl) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ku) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldafb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n_err_bnds) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nparams) > std::numeric_limits<lapack_int>::max() );
    }
    char trans_ = op2char( trans );
    char equed_ = equed2char( equed );
    lapack_int n_ = (lapack_int) n;
    lapack_int kl_ = (lapack_int) kl;
    lapack_int ku_ = (lapack_int) ku;
    lapack_int nrhs_ = (lapack_int) nrhs;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int ldafb_ = (lapack_int) ldafb;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int const* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int const* ipiv_ptr = ipiv;
    #endif
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int ldx_ = (lapack_int) ldx;
    lapack_int n_err_bnds_ = (lapack_int) n_err_bnds;
    lapack_int nparams_ = (lapack_int) nparams;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<float> > work( (2*n) );
    lapack::vector< float > rwork( (2*n) );

    LAPACK_cgbrfsx(
        &trans_, &equed_, &n_, &kl_, &ku_, &nrhs_,
        (lapack_complex_float*) AB, &ldab_,
        (lapack_complex_float*) AFB, &ldafb_,
        ipiv_ptr,
        R,
        C,
        (lapack_complex_float*) B, &ldb_,
        (lapack_complex_float*) X, &ldx_, rcond,
        berr, &n_err_bnds_,
        err_bnds_norm,
        err_bnds_comp, &nparams_,
        params,
        (lapack_complex_float*) &work[0],
        &rwork[0], &info_
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
/// Improves the computed solution to a system of linear
/// equations and provides error bounds and backward error estimates
/// for the solution. In addition to normwise error bound, the code
/// provides maximum componentwise error bound if possible. See
/// comments for err_bnds_norm and err_bnds_comp for details of the
/// error bounds.
///
/// The original system of linear equations may have been equilibrated
/// before calling this routine, as described by arguments equed, R
/// and C below. In this case, the solution and error bounds returned
/// are for the original unequilibrated system.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] trans
///     The form of the system of equations:
///     - lapack::Op::NoTrans:   $A   X = B$ (No transpose)
///     - lapack::Op::Trans:     $A^T X = B$ (Transpose)
///     - lapack::Op::ConjTrans: $A^H X = B$ (Conjugate transpose)
///
/// @param[in] equed
///     The form of equilibration that was done to A
///     before calling this routine. This is needed to compute
///     the solution and error bounds correctly.
///     - lapack::Equed::None: No equilibration
///     - lapack::Equed::Row:
///         Row equilibration, i.e., A has been premultiplied by diag(R).
///     - lapack::Equed::Col:
///         Column equilibration, i.e., A has been postmultiplied by diag(C).
///     - lapack::Equed::Both:
///         Both row and column equilibration, i.e.,
///         A has been replaced by $\text{diag}(R) \; A \; \text{diag}(C).$
///         The right hand side B has been changed accordingly.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] kl
///     The number of subdiagonals within the band of A. kl >= 0.
///
/// @param[in] ku
///     The number of superdiagonals within the band of A. ku >= 0.
///
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrices B and X. nrhs >= 0.
///
/// @param[in] AB
///     The n-by-n band matrix AB, stored in an ldab-by-n array.
///     The original band matrix A, stored in rows 1 to kl+ku+1.
///     The j-th column of A is stored in the j-th column of the
///     array AB as follows:
///     \n
///     AB(ku+1+i-j,j) = A(i,j) for max(1,j-ku) <= i <= min(n,j+kl).
///
/// @param[in] ldab
///     The leading dimension of the array AB. ldab >= kl+ku+1.
///
/// @param[in] AFB
///     The n-by-n band matrix AFB, stored in an ldafb-by-n array.
///     Details of the LU factorization of the band matrix A, as
///     computed by `lapack::gbtrf`. U is stored as an upper triangular band
///     matrix with kl+ku superdiagonals in rows 1 to kl+ku+1, and
///     the multipliers used during the factorization are stored in
///     rows kl+ku+2 to 2*kl+ku+1.
///
/// @param[in] ldafb
///     The leading dimension of the array AFB. ldafb >= 2*kl*ku+1.
///
/// @param[in] ipiv
///     The vector ipiv of length n.
///     The pivot indices from `lapack::getrf`; for 1 <= i <= n, row i of the
///     matrix was interchanged with row ipiv(i).
///
/// @param[in,out] R
///     The vector R of length n.
///     The row scale factors for A.
///     - If equed = Row or Both, A is multiplied on the left by diag(R);
///     otherwise, R is not accessed.
///
///     - If fact = Factored, R is an input argument;
///     otherwise, R is an output argument.
///
///     - If fact = Factored and
///     equed = Row or Both, each element of R must be positive.
///
///     - If R is output, each element of R is a power of the radix.
///     - If R is input, each element of R should be a power of the radix
///     to ensure a reliable solution and error estimates. Scaling by
///     powers of the radix does not cause rounding errors unless the
///     result underflows or overflows. Rounding errors during scaling
///     lead to refining with a matrix that is not equivalent to the
///     input matrix, producing error estimates that may not be reliable.
///
/// @param[in,out] C
///     The vector C of length n.
///     The column scale factors for A.
///     - If equed = Col or Both, A is multiplied on the right by diag(C);
///     otherwise, C is not accessed.
///
///     - If fact = Factored, C is an input argument;
///     otherwise, C is an output argument.
///
///     - If fact = Factored and
///     equed = Col or Both, each element of C must be positive.
///
///     - If C is output, each element of C is a power of the radix.
///     - If C is input, each element of C should be a power of the radix
///     to ensure a reliable solution and error estimates. Scaling by
///     powers of the radix does not cause rounding errors unless the
///     result underflows or overflows. Rounding errors during scaling
///     lead to refining with a matrix that is not equivalent to the
///     input matrix, producing error estimates that may not be reliable.
///
/// @param[in] B
///     The n-by-nrhs matrix B, stored in an ldb-by-nrhs array.
///     The right hand side matrix B.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,n).
///
/// @param[in,out] X
///     The n-by-nrhs matrix X, stored in an ldx-by-nrhs array.
///     On entry, the solution matrix X, as computed by `lapack::getrs`.
///     On exit, the improved solution matrix X.
///
/// @param[in] ldx
///     The leading dimension of the array X. ldx >= max(1,n).
///
/// @param[out] rcond
///     Reciprocal scaled condition number. This is an estimate of the
///     reciprocal Skeel condition number of the matrix A after
///     equilibration (if done). If this is less than the machine
///     precision (in particular, if it is zero), the matrix is singular
///     to working precision. Note that the error may still be small even
///     if this number is very small and the matrix appears ill-
///     conditioned.
///
/// @param[out] berr
///     The vector berr of length nrhs.
///     Componentwise relative backward error. This is the
///     componentwise relative backward error of each solution vector X(j)
///     (i.e., the smallest relative change in any element of A or B that
///     makes X(j) an exact solution).
///
/// @param[in] n_err_bnds
///     Number of error bounds to return for each right hand side
///     and each type (normwise or componentwise). See err_bnds_norm and
///     err_bnds_comp below.
///
/// @param[out] err_bnds_norm
///     An nrhs-by-n_err_bnds array.
///     For each right-hand side, this array contains information about
///     various error bounds and condition numbers corresponding to the
///     normwise relative error, which is defined as follows:
///     \n
///     Normwise relative error in the i-th solution vector:
///     \[
///         \frac{ \max_j | X_{true}(j,i) - X(j,i) | }
///              { \max_j | X(j,i) | }
///     \]
///     The array is indexed by the type of error information as described
///     below. There currently are up to three pieces of information
///     returned.
///     - The first index in err_bnds_norm(i,:) corresponds to the i-th
///     right-hand side.
///
///     - The second index in err_bnds_norm(:,err) contains the following
///     three fields:
///       - err = 1 "Trust/don't trust" boolean. Trust the answer if the
///               reciprocal condition number is less than the threshold
///               sqrt(n) * dlamch('Epsilon').
///
///       - err = 2 "Guaranteed" error bound: The estimated forward error,
///               almost certainly within a factor of 10 of the true error
///               so long as the next entry is greater than the threshold
///               sqrt(n) * dlamch('Epsilon'). This error bound should only
///               be trusted if the previous boolean is true.
///
///       - err = 3 Reciprocal condition number: Estimated normwise
///               reciprocal condition number. Compared with the threshold
///               sqrt(n) * dlamch('Epsilon') to determine if the error
///               estimate is "guaranteed". These reciprocal condition
///         numbers are $1 / (|| Z^{-1} ||_{inf} \cdot || Z ||_{inf})$ for some
///               appropriately scaled matrix Z.
///               Let $Z = S A,$ where S scales each row by a power of the
///               radix so all absolute row sums of Z are approximately 1.
///
///     - See Lapack Working Note 165 for further details and extra
///     cautions.
///
/// @param[out] err_bnds_comp
///     An nrhs-by-n_err_bnds array.
///     For each right-hand side, this array contains information about
///     various error bounds and condition numbers corresponding to the
///     componentwise relative error, which is defined as follows:
///     \n
///     Componentwise relative error in the i-th solution vector:
///     \[
///         \max_j \frac{ | X_{true}(j,i) - X(j,i) | }
///                     { | X(j,i) | }
///     \]
///     The array is indexed by the right-hand side i (on which the
///     componentwise relative error depends), and the type of error
///     information as described below. There currently are up to three
///     pieces of information returned for each right-hand side. If
///     componentwise accuracy is not requested (params(3) = 0.0), then
///     err_bnds_comp is not accessed. If n_err_bnds < 3, then at most
///     the first (:,n_err_bnds) entries are returned.
///     - The first index in err_bnds_comp(i,:) corresponds to the i-th
///     right-hand side.
///
///     - The second index in err_bnds_comp(:,err) contains the following
///     three fields:
///       - err = 1 "Trust/don't trust" boolean. Trust the answer if the
///           reciprocal condition number is less than the threshold
///           sqrt(n) * dlamch('Epsilon').
///
///       - err = 2 "Guaranteed" error bound: The estimated forward error,
///           almost certainly within a factor of 10 of the true error
///           so long as the next entry is greater than the threshold
///           sqrt(n) * dlamch('Epsilon'). This error bound should only
///           be trusted if the previous boolean is true.
///
///       - err = 3 Reciprocal condition number: Estimated componentwise
///           reciprocal condition number. Compared with the threshold
///           sqrt(n) * dlamch('Epsilon') to determine if the error
///           estimate is "guaranteed". These reciprocal condition
///         numbers are $1 / (|| Z^{-1} ||_{inf} \cdot || Z ||_{inf})$ for some
///           appropriately scaled matrix Z.
///         Let $Z = S A \; \text{diag}(x),$ where x is the solution for the
///           current right-hand side and S scales each row of
///         $A \; \text{diag}(x)$ by a power of the radix so all absolute row
///           sums of Z are approximately 1.
///
///     - See Lapack Working Note 165 for further details and extra
///     cautions.
///
/// @param[in] nparams
///     The number of parameters set in params. If <= 0, the
///     params array is never referenced and default values are used.
///
/// @param[in,out] params
///     The vector params of length nparams.
///     Algorithm parameters. If an entry is < 0.0, then
///     that entry will be filled with the default value used for that
///     parameter. Only positions up to nparams are accessed; defaults
///     are used for higher-numbered parameters.
///     - params(LA_LINRX_ITREF_I = 1):
///       Whether to perform iterative refinement or not.
///       - Default: 1.0
///       - 0.0 : No refinement is performed, and no error bounds are
///               computed.
///       - 1.0 : Use the double-precision refinement algorithm,
///               possibly with doubled-single computations if the
///               compilation environment does not support double precision.
///       - (other values are reserved for future use)
///
///     - params(LA_LINRX_ITHRESH_I = 2):
///       Maximum number of residual computations allowed for refinement.
///       - Default: 10
///       - Aggressive: Set to 100 to permit convergence using approximate
///                factorizations or factorizations other than LU. If
///                the factorization uses a technique other than
///                Gaussian elimination, the guarantees in
///                err_bnds_norm and err_bnds_comp may no longer be
///                trustworthy.
///
///     - params(LA_LINRX_CWISE_I = 3):
///       Flag determining if the code
///       will attempt to find a solution with small componentwise
///       relative error in the double-precision algorithm.
///       - Positive is true
///       - 0.0 is false
///       - Default: 1.0 (attempt componentwise convergence)
///
/// @return = 0: Successful exit.
///     The solution to every right-hand side is guaranteed.
/// @return > 0 and <= n: if return value = i,
///     U(i,i) is exactly zero. The factorization
///     has been completed, but the factor U is exactly singular, so
///     the solution and error bounds could not be computed. rcond = 0
///     is returned.
/// @return > n: if return value = n+j,
///     the solution corresponding to the j-th right-hand side is
///     not guaranteed. The solutions corresponding to other right-hand
///     sides k with k > j may not be guaranteed as well, but
///     only the first such right-hand side is reported. If a small
///     componentwise error is not requested (params(3) = 0.0) then
///     the j-th right-hand side is the first with a normwise error
///     bound that is not guaranteed (the smallest j such
///     that err_bnds_norm(j,1) = 0.0). By default (params(3) = 1.0)
///     the j-th right-hand side is the first with either a normwise or
///     componentwise error bound that is not guaranteed (the smallest
///     j such that either err_bnds_norm(j,1) = 0.0 or
///     err_bnds_comp(j,1) = 0.0). See the definition of
///     err_bnds_norm(:,1) and err_bnds_comp(:,1). To get information
///     about all of the right-hand sides check err_bnds_norm or
///     err_bnds_comp.
///
/// @ingroup gbsv_computational
int64_t gbrfsx(
    lapack::Op trans, lapack::Equed equed, int64_t n, int64_t kl, int64_t ku, int64_t nrhs,
    std::complex<double> const* AB, int64_t ldab,
    std::complex<double> const* AFB, int64_t ldafb,
    int64_t const* ipiv,
    double* R,
    double* C,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double>* X, int64_t ldx,
    double* rcond,
    double* berr, int64_t n_err_bnds,
    double* err_bnds_norm,
    double* err_bnds_comp, int64_t nparams,
    double* params )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kl) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ku) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldafb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n_err_bnds) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(nparams) > std::numeric_limits<lapack_int>::max() );
    }
    char trans_ = op2char( trans );
    char equed_ = equed2char( equed );
    lapack_int n_ = (lapack_int) n;
    lapack_int kl_ = (lapack_int) kl;
    lapack_int ku_ = (lapack_int) ku;
    lapack_int nrhs_ = (lapack_int) nrhs;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int ldafb_ = (lapack_int) ldafb;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        lapack_int const* ipiv_ptr = &ipiv_[0];
    #else
        lapack_int const* ipiv_ptr = ipiv;
    #endif
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int ldx_ = (lapack_int) ldx;
    lapack_int n_err_bnds_ = (lapack_int) n_err_bnds;
    lapack_int nparams_ = (lapack_int) nparams;
    lapack_int info_ = 0;

    // allocate workspace
    lapack::vector< std::complex<double> > work( (2*n) );
    lapack::vector< double > rwork( (2*n) );

    LAPACK_zgbrfsx(
        &trans_, &equed_, &n_, &kl_, &ku_, &nrhs_,
        (lapack_complex_double*) AB, &ldab_,
        (lapack_complex_double*) AFB, &ldafb_,
        ipiv_ptr,
        R,
        C,
        (lapack_complex_double*) B, &ldb_,
        (lapack_complex_double*) X, &ldx_, rcond,
        berr, &n_err_bnds_,
        err_bnds_norm,
        err_bnds_comp, &nparams_,
        params,
        (lapack_complex_double*) &work[0],
        &rwork[0], &info_
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

#endif  // HAVE_XBLAS
