#include "lapack.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup posv_computational
int64_t pocon(
    lapack::Uplo uplo, int64_t n,
    float const* A, int64_t lda, float anorm,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (3*n) );
    std::vector< lapack_int > iwork( (n) );

    LAPACK_spocon(
        &uplo_, &n_,
        A, &lda_, &anorm, rcond,
        &work[0],
        &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup posv_computational
int64_t pocon(
    lapack::Uplo uplo, int64_t n,
    double const* A, int64_t lda, double anorm,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (3*n) );
    std::vector< lapack_int > iwork( (n) );

    LAPACK_dpocon(
        &uplo_, &n_,
        A, &lda_, &anorm, rcond,
        &work[0],
        &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup posv_computational
int64_t pocon(
    lapack::Uplo uplo, int64_t n,
    std::complex<float> const* A, int64_t lda, float anorm,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (2*n) );
    std::vector< float > rwork( (n) );

    LAPACK_cpocon(
        &uplo_, &n_,
        (lapack_complex_float*) A, &lda_, &anorm, rcond,
        (lapack_complex_float*) &work[0],
        &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Estimates the reciprocal of the condition number (in the
/// 1-norm) of a Hermitian positive definite matrix using the
/// Cholesky factorization \f$ A = U^H U \f$ or \f$ A = L L^H \f$ computed by `lapack::potrf`.
///
/// An estimate is obtained for norm(inv(A)), and the reciprocal of the
/// condition number is computed as rcond = 1 / (anorm * norm(inv(A))).
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangle of A is stored;
///     - lapack::Uplo::Lower: Lower triangle of A is stored.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     The triangular factor U or L from the Cholesky factorization
///     \f$ A = U^H U \f$ or \f$ A = L L^H \f$, as computed by `lapack::potrf`.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[in] anorm
///     The 1-norm (or infinity-norm) of the Hermitian matrix A.
///
/// @param[out] rcond
///     The reciprocal of the condition number of the matrix A,
///     computed as rcond = 1/(anorm * ainv_norm), where ainv_norm is an
///     estimate of the 1-norm of inv(A) computed in this routine.
///
/// @retval = 0: successful exit
///
/// @ingroup posv_computational
int64_t pocon(
    lapack::Uplo uplo, int64_t n,
    std::complex<double> const* A, int64_t lda, double anorm,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (2*n) );
    std::vector< double > rwork( (n) );

    LAPACK_zpocon(
        &uplo_, &n_,
        (lapack_complex_double*) A, &lda_, &anorm, rcond,
        (lapack_complex_double*) &work[0],
        &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
