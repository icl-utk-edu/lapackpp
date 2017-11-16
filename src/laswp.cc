#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
void laswp(
    int64_t n,
    float* A, int64_t lda, int64_t k1, int64_t k2,
    int64_t const* ipiv, int64_t incx )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k1) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k2) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
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

    LAPACK_slaswp( &n_, A, &lda_, &k1_, &k2_, ipiv_ptr, &incx_ );
}

// -----------------------------------------------------------------------------
void laswp(
    int64_t n,
    double* A, int64_t lda, int64_t k1, int64_t k2,
    int64_t const* ipiv, int64_t incx )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k1) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k2) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
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

    LAPACK_dlaswp( &n_, A, &lda_, &k1_, &k2_, ipiv_ptr, &incx_ );
}

// -----------------------------------------------------------------------------
void laswp(
    int64_t n,
    std::complex<float>* A, int64_t lda, int64_t k1, int64_t k2,
    int64_t const* ipiv, int64_t incx )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k1) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k2) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
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

    LAPACK_claswp( &n_, A, &lda_, &k1_, &k2_, ipiv_ptr, &incx_ );
}

// -----------------------------------------------------------------------------
void laswp(
    int64_t n,
    std::complex<double>* A, int64_t lda, int64_t k1, int64_t k2,
    int64_t const* ipiv, int64_t incx )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k1) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k2) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
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

    LAPACK_zlaswp( &n_, A, &lda_, &k1_, &k2_, ipiv_ptr, &incx_ );
}

}  // namespace lapack
