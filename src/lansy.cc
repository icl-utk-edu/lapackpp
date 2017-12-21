#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup norm
float lansy(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    float const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    std::vector< float > work( max( 1, lwork ) );

    return LAPACK_slansy( &norm_, &uplo_, &n_, A, &lda_, &work[0] );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
double lansy(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    double const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    std::vector< double > work( max( 1, lwork ) );

    return LAPACK_dlansy( &norm_, &uplo_, &n_, A, &lda_, &work[0] );
}

// -----------------------------------------------------------------------------
/// @ingroup norm
float lansy(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    std::complex<float> const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    std::vector< float > work( max( 1, lwork ) );

    return LAPACK_clansy( &norm_, &uplo_, &n_, A, &lda_, &work[0] );
}

// -----------------------------------------------------------------------------
/// Returns the value of the one norm, Frobenius norm,
/// infinity norm, or the element of largest absolute value of a
/// complex symmetric matrix A.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, `lapack::lanhe` is an alias for this.
/// For complex Hermitian matrices, see `lapack::lanhe`.
///
/// @param[in] norm
///     The value to be returned:
///     - lapack::Norm::Max: max norm: max(abs(A(i,j))).
///                          Note this is not a consistent matrix norm.
///     - lapack::Norm::One: one norm: maximum column sum
///     - lapack::Norm::Inf: infinity norm: maximum row sum
///     - lapack::Norm::Fro: Frobenius norm: square root of sum of squares
///
/// @param[in] uplo
///     Whether the upper or lower triangular part of the
///     symmetric matrix A is to be referenced.
///     - lapack::Uplo::Upper: Upper triangular part of A is referenced
///     - lapack::Uplo::Lower: Lower triangular part of A is referenced
///
/// @param[in] n
///     The order of the matrix A. n >= 0. When n = 0, returns zero.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     The symmetric matrix A.
///     - If uplo = Upper, the leading n-by-n
///     upper triangular part of A contains the upper triangular part
///     of the matrix A, and the strictly lower triangular part of A
///     is not referenced.
///
///     - If uplo = Lower, the leading n-by-n lower
///     triangular part of A contains the lower triangular part of
///     the matrix A, and the strictly upper triangular part of A is
///     not referenced.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(n,1).
///
/// @ingroup norm
double lansy(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    std::complex<double> const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    std::vector< double > work( max( 1, lwork ) );

    return LAPACK_zlansy( &norm_, &uplo_, &n_, A, &lda_, &work[0] );
}

}  // namespace lapack
