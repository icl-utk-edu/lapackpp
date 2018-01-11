#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup ppsv_computational
int64_t ppequ(
    lapack::Uplo uplo, int64_t n,
    float const* AP,
    float* S,
    float* scond,
    float* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_sppequ( &uplo_, &n_, AP, S, scond, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ppsv_computational
int64_t ppequ(
    lapack::Uplo uplo, int64_t n,
    double const* AP,
    double* S,
    double* scond,
    double* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_dppequ( &uplo_, &n_, AP, S, scond, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ppsv_computational
int64_t ppequ(
    lapack::Uplo uplo, int64_t n,
    std::complex<float> const* AP,
    float* S,
    float* scond,
    float* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_cppequ( &uplo_, &n_, AP, S, scond, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes row and column scalings intended to equilibrate a
/// Hermitian positive definite matrix A in packed storage and reduce
/// its condition number (with respect to the two-norm). S contains the
/// scale factors, \f$ S_i = 1 / \sqrt{ A_{i,i} }, \f$ chosen so that the scaled matrix
/// B with elements \f$ B_{i,j} = S_{i} A_{i,j} S_{j} \f$ has ones on the diagonal.
/// This choice of S puts the condition number of B within a factor n of
/// the smallest possible condition number over all possible diagonal
/// scalings.
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
///     The upper or lower triangle of the Hermitian matrix A, packed
///     columnwise in a linear array. The j-th column of A is stored
///     in the array AP as follows:
///     - if uplo = Upper, AP(i + (j-1)*j/2) = A(i,j) for 1 <= i <= j;
///     - if uplo = Lower, AP(i + (j-1)*(2n-j)/2) = A(i,j) for j <= i <= n.
///
/// @param[out] S
///     The vector S of length n.
///     If successful, S contains the scale factors for A.
///
/// @param[out] scond
///     If successful, S contains the ratio of the smallest S(i) to
///     the largest S(i). If scond >= 0.1 and amax is neither too
///     large nor too small, it is not worth scaling by S.
///
/// @param[out] amax
///     Absolute value of largest matrix element. If amax is very
///     close to overflow or very close to underflow, the matrix
///     should be scaled.
///
/// @retval = 0: successful exit
/// @retval > 0: if return value = i, the i-th diagonal element is nonpositive.
///
/// @ingroup ppsv_computational
int64_t ppequ(
    lapack::Uplo uplo, int64_t n,
    std::complex<double> const* AP,
    double* S,
    double* scond,
    double* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_zppequ( &uplo_, &n_, AP, S, scond, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
