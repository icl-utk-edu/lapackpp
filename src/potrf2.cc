#include "lapack.hh"
#include "lapack/fortran.h"

#if LAPACK_VERSION >= 30600  // >= v3.6

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup posv_computational
int64_t potrf2(
    lapack::Uplo uplo, int64_t n,
    float* A, int64_t lda )
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

    LAPACK_spotrf2(
        &uplo_, &n_,
        A, &lda_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup posv_computational
int64_t potrf2(
    lapack::Uplo uplo, int64_t n,
    double* A, int64_t lda )
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

    LAPACK_dpotrf2(
        &uplo_, &n_,
        A, &lda_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup posv_computational
int64_t potrf2(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda )
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

    LAPACK_cpotrf2(
        &uplo_, &n_,
        (lapack_complex_float*) A, &lda_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the Cholesky factorization of a Hermitian
/// positive definite matrix A using the recursive algorithm.
///
/// The factorization has the form
///     \f$ A = U^H U, \f$ if uplo = Upper, or
///     \f$ A = L L^H, \f$ if uplo = Lower,
/// where U is an upper triangular matrix and L is lower triangular.
///
/// This is the recursive version of the algorithm. It divides
/// the matrix into four submatrices:
/**
    \f[
        A = \left[ \begin{array}{cc}
            A_{11}  &  A_{12}  \\
            A_{21}  &  A_{22}  \\
        \end{array} \right]
    \f]
    where \f$ A_{11} \f$ is n1-by-n1 and \f$ A_{22} \f$ is n2-by-n2,
    with n1 = n/2 and n2 = n-n1.
    The subroutine calls itself to factor \f$ A_{11}, \f$
    updates and scales \f$ A_{21} \f$ or \f$ A_{12}, \f$
    updates \f$ A_{22}, \f$
    and calls itself to factor \f$ A_{22}. \f$
*/
///
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
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     On entry, the Hermitian matrix A.
///     - If uplo = Upper, the leading
///     n-by-n upper triangular part of A contains the upper
///     triangular part of the matrix A, and the strictly lower
///     triangular part of A is not referenced.
///
///     - If uplo = Lower, the
///     leading n-by-n lower triangular part of A contains the lower
///     triangular part of the matrix A, and the strictly upper
///     triangular part of A is not referenced.
///
///     - On successful exit, the factor U or L from the Cholesky
///     factorization \f$ A = U^H U \f$ or \f$ A = L L^H. \f$
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @retval = 0: successful exit
/// @retval > 0: if return value = i, the leading minor of order i is not
///     positive definite, and the factorization could not be
///     completed.
///
/// @ingroup posv_computational
int64_t potrf2(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda )
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

    LAPACK_zpotrf2(
        &uplo_, &n_,
        (lapack_complex_double*) A, &lda_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= v3.6
