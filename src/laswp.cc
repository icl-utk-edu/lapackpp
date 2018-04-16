#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gesv_computational
void laswp(
    int64_t n,
    float* A, int64_t lda, int64_t k1, int64_t k2,
    int64_t const* ipiv, int64_t incx )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k1) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k2) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int k1_ = (blas_int) k1;
    blas_int k2_ = (blas_int) k2;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(k1+(k2-k1)*abs(incx))] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int incx_ = (blas_int) incx;

    LAPACK_slaswp(
        &n_,
        A, &lda_, &k1_, &k2_,
        ipiv_ptr, &incx_ );
}

// -----------------------------------------------------------------------------
/// @ingroup gesv_computational
void laswp(
    int64_t n,
    double* A, int64_t lda, int64_t k1, int64_t k2,
    int64_t const* ipiv, int64_t incx )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k1) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k2) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int k1_ = (blas_int) k1;
    blas_int k2_ = (blas_int) k2;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(k1+(k2-k1)*abs(incx))] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int incx_ = (blas_int) incx;

    LAPACK_dlaswp(
        &n_,
        A, &lda_, &k1_, &k2_,
        ipiv_ptr, &incx_ );
}

// -----------------------------------------------------------------------------
/// @ingroup gesv_computational
void laswp(
    int64_t n,
    std::complex<float>* A, int64_t lda, int64_t k1, int64_t k2,
    int64_t const* ipiv, int64_t incx )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k1) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k2) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int k1_ = (blas_int) k1;
    blas_int k2_ = (blas_int) k2;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(k1+(k2-k1)*abs(incx))] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int incx_ = (blas_int) incx;

    LAPACK_claswp(
        &n_,
        (lapack_complex_float*) A, &lda_, &k1_, &k2_,
        ipiv_ptr, &incx_ );
}

// -----------------------------------------------------------------------------
/// Performs a series of row interchanges on the matrix A.
/// One row interchange is initiated for each of rows k1 through k2 of A.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] n
///     The number of columns of the matrix A.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     On entry, the matrix of column dimension n to which the row
///     interchanges will be applied.
///     On exit, the permuted matrix.
///     Note that the number of rows, m, is implicit in ipiv; m <= lda.
///
/// @param[in] lda
///     The leading dimension of the array A.
///
/// @param[in] k1
///     The first element of ipiv for which a row interchange will
///     be done.
///
/// @param[in] k2
///     (k2-k1+1) is the number of elements of ipiv for which a row
///     interchange will be done.
///
/// @param[in] ipiv
///     The vector ipiv of length k1+(k2-k1)*abs(incx).
///     The vector of pivot indices. Only the elements in positions
///     k1 through k1+(k2-k1)*abs(incx) of ipiv are accessed.
///     ipiv(k1+(\f$ K- \f$k1)*abs(incx)) = L implies rows K and L are to be
///     interchanged.
///
/// @param[in] incx
///     The increment between successive values of ipiv. If incx
///     is negative, the pivots are applied in reverse order.
///
/// @ingroup gesv_computational
void laswp(
    int64_t n,
    std::complex<double>* A, int64_t lda, int64_t k1, int64_t k2,
    int64_t const* ipiv, int64_t incx )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k1) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k2) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int k1_ = (blas_int) k1;
    blas_int k2_ = (blas_int) k2;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(k1+(k2-k1)*abs(incx))] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int incx_ = (blas_int) incx;

    LAPACK_zlaswp(
        &n_,
        (lapack_complex_double*) A, &lda_, &k1_, &k2_,
        ipiv_ptr, &incx_ );
}

}  // namespace lapack
