#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t sytri2x(
    lapack::Uplo uplo, int64_t n,
    float* A, int64_t lda,
    int64_t const* ipiv, int64_t nb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nb) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int nb_ = (blas_int) nb;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (n+nb+1)*(nb+3) );

    LAPACK_ssytri2x( &uplo_, &n_, A, &lda_, ipiv_ptr, &work[0], &nb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t sytri2x(
    lapack::Uplo uplo, int64_t n,
    double* A, int64_t lda,
    int64_t const* ipiv, int64_t nb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nb) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int nb_ = (blas_int) nb;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (n+nb+1)*(nb+3) );

    LAPACK_dsytri2x( &uplo_, &n_, A, &lda_, ipiv_ptr, &work[0], &nb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t sytri2x(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda,
    int64_t const* ipiv, int64_t nb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nb) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int nb_ = (blas_int) nb;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (n+nb+1)*(nb+3) );

    LAPACK_csytri2x( &uplo_, &n_, A, &lda_, ipiv_ptr, &work[0], &nb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t sytri2x(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda,
    int64_t const* ipiv, int64_t nb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nb) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int nb_ = (blas_int) nb;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (n+nb+1)*(nb+3) );

    LAPACK_zsytri2x( &uplo_, &n_, A, &lda_, ipiv_ptr, &work[0], &nb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
