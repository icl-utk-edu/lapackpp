#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t pstrf(
    lapack::Uplo uplo, int64_t n,
    float* A, int64_t lda,
    int64_t* piv,
    int64_t* rank, float tol )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > piv_( (n) );
        blas_int* piv_ptr = &piv_[0];
    #else
        blas_int* piv_ptr = piv;
    #endif
    blas_int rank_ = (blas_int) *rank;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (2*n) );

    LAPACK_spstrf( &uplo_, &n_, A, &lda_, piv_ptr, &rank_, &tol, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( piv_.begin(), piv_.end(), piv );
    #endif
    *rank = rank_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t pstrf(
    lapack::Uplo uplo, int64_t n,
    double* A, int64_t lda,
    int64_t* piv,
    int64_t* rank, double tol )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > piv_( (n) );
        blas_int* piv_ptr = &piv_[0];
    #else
        blas_int* piv_ptr = piv;
    #endif
    blas_int rank_ = (blas_int) *rank;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (2*n) );

    LAPACK_dpstrf( &uplo_, &n_, A, &lda_, piv_ptr, &rank_, &tol, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( piv_.begin(), piv_.end(), piv );
    #endif
    *rank = rank_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t pstrf(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda,
    int64_t* piv,
    int64_t* rank, float tol )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > piv_( (n) );
        blas_int* piv_ptr = &piv_[0];
    #else
        blas_int* piv_ptr = piv;
    #endif
    blas_int rank_ = (blas_int) *rank;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (2*n) );

    LAPACK_cpstrf( &uplo_, &n_, A, &lda_, piv_ptr, &rank_, &tol, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( piv_.begin(), piv_.end(), piv );
    #endif
    *rank = rank_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t pstrf(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda,
    int64_t* piv,
    int64_t* rank, double tol )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > piv_( (n) );
        blas_int* piv_ptr = &piv_[0];
    #else
        blas_int* piv_ptr = piv;
    #endif
    blas_int rank_ = (blas_int) *rank;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (2*n) );

    LAPACK_zpstrf( &uplo_, &n_, A, &lda_, piv_ptr, &rank_, &tol, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( piv_.begin(), piv_.end(), piv );
    #endif
    *rank = rank_;
    return info_;
}

}  // namespace lapack
