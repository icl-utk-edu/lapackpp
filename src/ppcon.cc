#include "lapack.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup ppsv_computational
int64_t ppcon(
    lapack::Uplo uplo, int64_t n,
    float const* AP, float anorm,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (3*n) );
    std::vector< lapack_int > iwork( (n) );

    LAPACK_sppcon(
        &uplo_, &n_,
        AP, &anorm, rcond,
        &work[0],
        &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ppsv_computational
int64_t ppcon(
    lapack::Uplo uplo, int64_t n,
    double const* AP, double anorm,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (3*n) );
    std::vector< lapack_int > iwork( (n) );

    LAPACK_dppcon(
        &uplo_, &n_,
        AP, &anorm, rcond,
        &work[0],
        &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ppsv_computational
int64_t ppcon(
    lapack::Uplo uplo, int64_t n,
    std::complex<float> const* AP, float anorm,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (2*n) );
    std::vector< float > rwork( (n) );

    LAPACK_cppcon(
        &uplo_, &n_,
        (lapack_complex_float*) AP, &anorm, rcond,
        (lapack_complex_float*) &work[0],
        &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Estimates the reciprocal of the condition number (in the
/// 1-norm) of a Hermitian positive definite packed matrix using
/// the Cholesky factorization \f$ A = U^H U \f$ or \f$ A = L L^H \f$ computed by
/// `lapack::pptrf`.
///
/// An estimate is obtained for \f$ ||A^{-1}||_1 \f$, and the reciprocal of the
/// condition number is computed as \f$ \text{rcond} = 1 / (||A||_1 \cdot ||A^{-1}||_1). \f$
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
/// @param[in] AP
///     The vector AP of length n*(n+1)/2.
///     The triangular factor U or L from the Cholesky factorization
///     \f$ A = U^H U \f$ or \f$ A = L L^H, \f$ packed columnwise in a linear
///     array. The j-th column of U or L is stored in the array AP
///     as follows:
///     - if uplo = Upper, AP(i + (j-1)*j/2) = U(i,j) for 1 <= i <= j;
///     - if uplo = Lower, AP(i + (j-1)*(2n-j)/2) = L(i,j) for j <= i <= n.
///
/// @param[in] anorm
///     The 1-norm (or infinity-norm) of the Hermitian matrix A.
///
/// @param[out] rcond
///     The reciprocal of the condition number of the matrix A,
///     computed as rcond = 1/(anorm * ainv_norm), where ainv_norm is an
///     estimate of the 1-norm of \f$ A^{-1} \f$ computed in this routine.
///
/// @retval = 0: successful exit
///
/// @ingroup ppsv_computational
int64_t ppcon(
    lapack::Uplo uplo, int64_t n,
    std::complex<double> const* AP, double anorm,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (2*n) );
    std::vector< double > rwork( (n) );

    LAPACK_zppcon(
        &uplo_, &n_,
        (lapack_complex_double*) AP, &anorm, rcond,
        (lapack_complex_double*) &work[0],
        &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
