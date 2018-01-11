#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gtsv_computational
int64_t gtcon(
    lapack::Norm norm, int64_t n,
    float const* DL,
    float const* D,
    float const* DU,
    float const* DU2,
    int64_t const* ipiv, float anorm,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    blas_int n_ = (blas_int) n;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (2*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_sgtcon( &norm_, &n_, DL, D, DU, DU2, ipiv_ptr, &anorm, rcond, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gtsv_computational
int64_t gtcon(
    lapack::Norm norm, int64_t n,
    double const* DL,
    double const* D,
    double const* DU,
    double const* DU2,
    int64_t const* ipiv, double anorm,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    blas_int n_ = (blas_int) n;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (2*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_dgtcon( &norm_, &n_, DL, D, DU, DU2, ipiv_ptr, &anorm, rcond, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gtsv_computational
int64_t gtcon(
    lapack::Norm norm, int64_t n,
    std::complex<float> const* DL,
    std::complex<float> const* D,
    std::complex<float> const* DU,
    std::complex<float> const* DU2,
    int64_t const* ipiv, float anorm,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    blas_int n_ = (blas_int) n;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (2*n) );

    LAPACK_cgtcon( &norm_, &n_, DL, D, DU, DU2, ipiv_ptr, &anorm, rcond, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Estimates the reciprocal of the condition number of a complex
/// tridiagonal matrix A using the LU factorization as computed by
/// `lapack::gttrf`.
///
/// An estimate is obtained for \f$ || A^{-1} ||, \f$ and the reciprocal of the
/// condition number is computed as  \f$ \text{rcond} = 1 / (|| A || \cdot || A^{-1} ||). \f$
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] norm
///     Specifies whether the 1-norm condition number or the
///     infinity-norm condition number is required:
///     - lapack::Norm::One: 1-norm;
///     - lapack::Norm::Inf: Infinity-norm.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] DL
///     The vector DL of length n-1.
///     The (n-1) multipliers that define the matrix L from the
///     LU factorization of A as computed by `lapack::gttrf`.
///
/// @param[in] D
///     The vector D of length n.
///     The n diagonal elements of the upper triangular matrix U from
///     the LU factorization of A.
///
/// @param[in] DU
///     The vector DU of length n-1.
///     The (n-1) elements of the first superdiagonal of U.
///
/// @param[in] DU2
///     The vector DU2 of length n-2.
///     The (n-2) elements of the second superdiagonal of U.
///
/// @param[in] ipiv
///     The vector ipiv of length n.
///     The pivot indices; for 1 <= i <= n, row i of the matrix was
///     interchanged with row ipiv(i). ipiv(i) will always be either
///     i or i+1; ipiv(i) = i indicates a row interchange was not
///     required.
///
/// @param[in] anorm
///     - If norm = One, the 1-norm of the original matrix A.
///     - If norm = Inf, the infinity-norm of the original matrix A.
///
/// @param[out] rcond
///     The reciprocal of the condition number of the matrix A,
///     computed as rcond = 1/(anorm * ainv_norm), where ainv_norm is an
///     estimate of the 1-norm or infinity-norm of \f$ A^{-1} \f$ computed in this routine.
///
/// @retval = 0: successful exit
///
/// @ingroup gtsv_computational
int64_t gtcon(
    lapack::Norm norm, int64_t n,
    std::complex<double> const* DL,
    std::complex<double> const* D,
    std::complex<double> const* DU,
    std::complex<double> const* DU2,
    int64_t const* ipiv, double anorm,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    blas_int n_ = (blas_int) n;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (2*n) );

    LAPACK_zgtcon( &norm_, &n_, DL, D, DU, DU2, ipiv_ptr, &anorm, rcond, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
