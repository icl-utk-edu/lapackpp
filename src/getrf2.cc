#include "lapack.hh"
#include "lapack_fortran.h"

#if LAPACK_VERSION_MAJOR >= 3 && LAPACK_VERSION_MINOR >= 6  // >= v3.6

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gesv_computational
int64_t getrf2(
    int64_t m, int64_t n,
    float* A, int64_t lda,
    int64_t* ipiv )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( max( 1, min( m, n )) );
        blas_int* ipiv_ptr = &ipiv_[0];
    #else
        blas_int* ipiv_ptr = ipiv;
    #endif
    blas_int info_ = 0;

    LAPACK_sgetrf2( &m_, &n_, A, &lda_, ipiv_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesv_computational
int64_t getrf2(
    int64_t m, int64_t n,
    double* A, int64_t lda,
    int64_t* ipiv )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( max( 1, min( m, n )) );
        blas_int* ipiv_ptr = &ipiv_[0];
    #else
        blas_int* ipiv_ptr = ipiv;
    #endif
    blas_int info_ = 0;

    LAPACK_dgetrf2( &m_, &n_, A, &lda_, ipiv_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gesv_computational
int64_t getrf2(
    int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda,
    int64_t* ipiv )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( max( 1, min( m, n )) );
        blas_int* ipiv_ptr = &ipiv_[0];
    #else
        blas_int* ipiv_ptr = ipiv;
    #endif
    blas_int info_ = 0;

    LAPACK_cgetrf2( &m_, &n_, A, &lda_, ipiv_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes an LU factorization of a general m-by-n matrix A
/// using partial pivoting with row interchanges.
///
/// The factorization has the form
///     \f$ A = P L U \f$
/// where P is a permutation matrix, L is lower triangular with unit
/// diagonal elements (lower trapezoidal if m > n), and U is upper
/// triangular (upper trapezoidal if m < n).
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
    with n1 = min(m,n)/2 and n2 = n-n1.
    The subroutine calls itself to factor
    \f$
        \left[ \begin{array}{c}
            A_{11}  \\
            A_{21}  \\
        \end{array} \right],
    \f$
    does the swaps on
    \f$
        \left[ \begin{array}{c}
            A_{12}  \\
            A_{22}  \\
        \end{array} \right],
    \f$
    solves \f$ A_{12}, \f$
    updates \f$ A_{22}, \f$
    calls itself to factor \f$ A_{22}, \f$
    and does the swaps on \f$ A_{21}. \f$
*/
///
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] m
///     The number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix A. n >= 0.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     On entry, the m-by-n matrix to be factored.
///     On exit, the factors L and U from the factorization
///     \f$ A = P L U; \f$ the unit diagonal elements of L are not stored.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @param[out] ipiv
///     The vector ipiv of length min(m,n).
///     The pivot indices; for 1 <= i <= min(m,n), row i of the
///     matrix was interchanged with row ipiv(i).
///
/// @retval = 0: successful exit
/// @retval > 0: if return value = i, U(i,i) is exactly zero. The factorization
///     has been completed, but the factor U is exactly
///     singular, and division by zero will occur if it is used
///     to solve a system of equations.
///
/// @ingroup gesv_computational
int64_t getrf2(
    int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda,
    int64_t* ipiv )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( (min(m,n)) );
        blas_int* ipiv_ptr = &ipiv_[0];
    #else
        blas_int* ipiv_ptr = ipiv;
    #endif
    blas_int info_ = 0;

    LAPACK_zgetrf2( &m_, &n_, A, &lda_, ipiv_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= v3.6
