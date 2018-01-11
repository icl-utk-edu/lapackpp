#include "lapack.hh"
#include "lapack_fortran.h"

#ifdef HAVE_XBLAS

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup sysv_computational
int64_t syrfsx(
    lapack::Uplo uplo, lapack::Equed equed, int64_t n, int64_t nrhs,
    float const* A, int64_t lda,
    float const* AF, int64_t ldaf,
    int64_t const* ipiv,
    float* S,
    float const* B, int64_t ldb,
    float* X, int64_t ldx,
    float* rcond,
    float* berr, int64_t n_err_bnds,
    float* err_bnds_norm,
    float* err_bnds_comp, int64_t nparams,
    float* params )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldaf) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n_err_bnds) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(nparams) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char equed_ = equed2char( equed );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    blas_int ldaf_ = (blas_int) ldaf;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int n_err_bnds_ = (blas_int) n_err_bnds;
    blas_int nparams_ = (blas_int) nparams;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (4*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_ssyrfsx( &uplo_, &equed_, &n_, &nrhs_, A, &lda_, AF, &ldaf_, ipiv_ptr, S, B, &ldb_, X, &ldx_, rcond, berr, &n_err_bnds_, err_bnds_norm, err_bnds_comp, &nparams_, params, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup sysv_computational
int64_t syrfsx(
    lapack::Uplo uplo, lapack::Equed equed, int64_t n, int64_t nrhs,
    double const* A, int64_t lda,
    double const* AF, int64_t ldaf,
    int64_t const* ipiv,
    double* S,
    double const* B, int64_t ldb,
    double* X, int64_t ldx,
    double* rcond,
    double* berr, int64_t n_err_bnds,
    double* err_bnds_norm,
    double* err_bnds_comp, int64_t nparams,
    double* params )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldaf) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n_err_bnds) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(nparams) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char equed_ = equed2char( equed );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    blas_int ldaf_ = (blas_int) ldaf;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int n_err_bnds_ = (blas_int) n_err_bnds;
    blas_int nparams_ = (blas_int) nparams;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (4*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_dsyrfsx( &uplo_, &equed_, &n_, &nrhs_, A, &lda_, AF, &ldaf_, ipiv_ptr, S, B, &ldb_, X, &ldx_, rcond, berr, &n_err_bnds_, err_bnds_norm, err_bnds_comp, &nparams_, params, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup sysv_computational
int64_t syrfsx(
    lapack::Uplo uplo, lapack::Equed equed, int64_t n, int64_t nrhs,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* AF, int64_t ldaf,
    int64_t const* ipiv,
    float* S,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float>* X, int64_t ldx,
    float* rcond,
    float* berr, int64_t n_err_bnds,
    float* err_bnds_norm,
    float* err_bnds_comp, int64_t nparams,
    float* params )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldaf) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n_err_bnds) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(nparams) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char equed_ = equed2char( equed );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    blas_int ldaf_ = (blas_int) ldaf;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int n_err_bnds_ = (blas_int) n_err_bnds;
    blas_int nparams_ = (blas_int) nparams;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (2*n) );
    std::vector< float > rwork( (2*n) );

    LAPACK_csyrfsx( &uplo_, &equed_, &n_, &nrhs_, A, &lda_, AF, &ldaf_, ipiv_ptr, S, B, &ldb_, X, &ldx_, rcond, berr, &n_err_bnds_, err_bnds_norm, err_bnds_comp, &nparams_, params, &work[0], &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Improves the computed solution to a system of linear
/// equations when the coefficient matrix is symmetric indefinite, and
/// provides error bounds and backward error estimates for the
/// solution. In addition to normwise error bound, the code provides
/// maximum componentwise error bound if possible. See comments for
/// err_bnds_norm and err_bnds_comp for details of the error bounds.
///
/// The original system of linear equations may have been equilibrated
/// before calling this routine, as described by arguments equed and S
/// below. In this case, the solution and error bounds returned are
/// for the original unequilibrated system.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, `lapack::herfsx` is an alias for this.
/// For complex Hermitian matrices, see `lapack::herfsx`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangle of A is stored;
///     - lapack::Uplo::Lower: Lower triangle of A is stored.
///
/// @param[in] equed
///     The form of equilibration that was done to A
///     before calling this routine. This is needed to compute
///     the solution and error bounds correctly.
///     - lapack::Equed::None: No equilibration
///     - lapack::Equed::Yes:
///         Both row and column equilibration, i.e.,
///         A has been replaced by \f$ \text{diag}(S) \; A \; \text{diag}(S). \f$
///         The right hand side B has been changed accordingly.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrices B and X. nrhs >= 0.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     The symmetric matrix A.
///     - If uplo = Upper, the leading n-by-n
///     upper triangular part of A contains the upper triangular
///     part of the matrix A, and the strictly lower triangular
///     part of A is not referenced.
///     - If uplo = Lower, the leading
///     n-by-n lower triangular part of A contains the lower
///     triangular part of the matrix A, and the strictly upper
///     triangular part of A is not referenced.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[in] AF
///     The n-by-n matrix AF, stored in an ldaf-by-n array.
///     The factored form of the matrix A. AF contains the block
///     diagonal matrix D and the multipliers used to obtain the
///     factor U or L from the factorization \f$ A = U D U^T \f$ or \f$ A = \f$
///     \f$ L D L^T \f$ as computed by `lapack::sytrf`.
///
/// @param[in] ldaf
///     The leading dimension of the array AF. ldaf >= max(1,n).
///
/// @param[in] ipiv
///     The vector ipiv of length n.
///     Details of the interchanges and the block structure of D
///     as determined by `lapack::sytrf`.
///
/// @param[in,out] S
///     The vector S of length n.
///     The scale factors for A.
///     - If equed = Yes, A is multiplied on the left and right by diag(S).
///
///     - If fact = Factored, S is an input argument. Each element
///     of S should be a power of the radix to ensure a reliable solution
///     and error estimates. Scaling by powers of the radix does not cause
///     rounding errors unless the result underflows or overflows.
///     Rounding errors during scaling lead to refining with a matrix that
///     is not equivalent to the input matrix, producing error estimates
///     that may not be reliable.
///
///     - otherwise, S is an output argument.
///     Each element of S is a power of the radix.
///
///     - If fact = Factored and equed = Yes, each element of S must be positive.
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
///     \f[
///         \frac{ \max_j | X_{true}(j,i) - X(j,i) | }
///              { \max_j | X(j,i) | }
///     \f]
///     The array is indexed by the type of error information as described
///     below. There currently are up to three pieces of information
///     returned.
///     - The first index in err_bnds_norm(i,:) corresponds to the i-th
///     right-hand side.
///
///     - The second index in err_bnds_norm(:,err) contains the following
///     three fields:
///       - err = 1 "Trust/don't trust" boolean. Trust the answer if the
///         reciprocal condition number is less than the threshold
///         sqrt(n) * dlamch('Epsilon').
///
///       - err = 2 "Guaranteed" error bound: The estimated forward error,
///         almost certainly within a factor of 10 of the true error
///         so long as the next entry is greater than the threshold
///         sqrt(n) * dlamch('Epsilon'). This error bound should only
///         be trusted if the previous boolean is true.
///
///       - err = 3 Reciprocal condition number: Estimated normwise
///         reciprocal condition number. Compared with the threshold
///         sqrt(n) * dlamch('Epsilon') to determine if the error
///         estimate is "guaranteed". These reciprocal condition
///         numbers are \f$ 1 / (|| Z^{-1} ||_{inf} \cdot || Z ||_{inf}) \f$ for some
///         appropriately scaled matrix Z.
///         Let \f$ Z = S A, \f$ where S scales each row by a power of the
///         radix so all absolute row sums of Z are approximately 1.
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
///     \f[
///         \max_j \frac{ | X_{true}(j,i) - X(j,i) | }
///                     { | X(j,i) | }
///     \f]
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
///         reciprocal condition number is less than the threshold
///         sqrt(n) * dlamch('Epsilon').
///
///       - err = 2 "Guaranteed" error bound: The estimated forward error,
///         almost certainly within a factor of 10 of the true error
///         so long as the next entry is greater than the threshold
///         sqrt(n) * dlamch('Epsilon'). This error bound should only
///         be trusted if the previous boolean is true.
///
///       - err = 3 Reciprocal condition number: Estimated componentwise
///         reciprocal condition number. Compared with the threshold
///         sqrt(n) * dlamch('Epsilon') to determine if the error
///         estimate is "guaranteed". These reciprocal condition
///         numbers are \f$ 1 / (|| Z^{-1} ||_{inf} \cdot || Z ||_{inf}) \f$ for some
///         appropriately scaled matrix Z.
///         Let \f$ Z = S A \; \text{diag}(x), \f$ where x is the solution for the
///         current right-hand side and S scales each row of
///         \f$ A \; \text{diag}(x) \f$ by a power of the radix so all absolute row
///         sums of Z are approximately 1.
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
///         computed.
///       - 1.0 : Use the double-precision refinement algorithm,
///         possibly with doubled-single computations if the
///         compilation environment does not support double precision.
///       - (other values are reserved for future use)
///
///     - params(LA_LINRX_ITHRESH_I = 2):
///       Maximum number of residual computations allowed for refinement.
///       - Default: 10
///       - Aggressive: Set to 100 to permit convergence using approximate
///         factorizations or factorizations other than LU. If
///         the factorization uses a technique other than
///         Gaussian elimination, the guarantees in
///         err_bnds_norm and err_bnds_comp may no longer be
///         trustworthy.
///
///     - params(LA_LINRX_CWISE_I = 3):
///       Flag determining if the code
///       will attempt to find a solution with small componentwise
///       relative error in the double-precision algorithm.
///       - Positive is true
///       - 0.0 is false
///       - Default: 1.0 (attempt componentwise convergence)
///
/// @retval = 0: Successful exit.
///     The solution to every right-hand side is guaranteed.
/// @retval > 0 and <= n: if return value = i,
///     U(i,i) is exactly zero. The factorization
///     has been completed, but the factor U is exactly singular, so
///     the solution and error bounds could not be computed. rcond = 0
///     is returned.
/// @retval > n: if return value = n+j,
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
/// @ingroup sysv_computational
int64_t syrfsx(
    lapack::Uplo uplo, lapack::Equed equed, int64_t n, int64_t nrhs,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* AF, int64_t ldaf,
    int64_t const* ipiv,
    double* S,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double>* X, int64_t ldx,
    double* rcond,
    double* berr, int64_t n_err_bnds,
    double* err_bnds_norm,
    double* err_bnds_comp, int64_t nparams,
    double* params )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldaf) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n_err_bnds) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(nparams) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char equed_ = equed2char( equed );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    blas_int ldaf_ = (blas_int) ldaf;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int n_err_bnds_ = (blas_int) n_err_bnds;
    blas_int nparams_ = (blas_int) nparams;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (2*n) );
    std::vector< double > rwork( (2*n) );

    LAPACK_zsyrfsx( &uplo_, &equed_, &n_, &nrhs_, A, &lda_, AF, &ldaf_, ipiv_ptr, S, B, &ldb_, X, &ldx_, rcond, berr, &n_err_bnds_, err_bnds_norm, err_bnds_comp, &nparams_, params, &work[0], &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // HAVE_XBLAS
