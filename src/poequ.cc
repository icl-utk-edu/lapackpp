#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup posv_computational
int64_t poequ(
    int64_t n,
    float const* A, int64_t lda,
    float* S,
    float* scond,
    float* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    LAPACK_spoequ( &n_, A, &lda_, S, scond, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup posv_computational
int64_t poequ(
    int64_t n,
    double const* A, int64_t lda,
    double* S,
    double* scond,
    double* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    LAPACK_dpoequ( &n_, A, &lda_, S, scond, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup posv_computational
int64_t poequ(
    int64_t n,
    std::complex<float> const* A, int64_t lda,
    float* S,
    float* scond,
    float* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    LAPACK_cpoequ( &n_, A, &lda_, S, scond, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes row and column scalings intended to equilibrate a
/// Hermitian positive definite matrix A and reduce its condition number
/// (with respect to the two-norm). S contains the scale factors,
/// \f$ S_{i} = 1/\sqrt{ A_{i,i} }, \f$ chosen so that the scaled matrix B with
/// elements \f$ B_{i,j} = S_{i} A_{i,j} S_{j} \f$ has ones on the diagonal. This
/// choice of S puts the condition number of B within a factor n of the
/// smallest possible condition number over all possible diagonal
/// scalings.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     The n-by-n Hermitian positive definite matrix whose scaling
///     factors are to be computed. Only the diagonal elements of A
///     are referenced.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
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
/// @ingroup posv_computational
int64_t poequ(
    int64_t n,
    std::complex<double> const* A, int64_t lda,
    double* S,
    double* scond,
    double* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    LAPACK_zpoequ( &n_, A, &lda_, S, scond, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
