#include "lapack.hh"
#include "lapack_fortran.h"

#ifdef LAPACK_MATGEN

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t laghe(
    int64_t n, int64_t k,
    float const* D,
    std::complex<float>* A, int64_t lda,
    int64_t* iseed )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > iseed_( &iseed[0], &iseed[(4)] );
        blas_int* iseed_ptr = &iseed_[0];
    #else
        blas_int* iseed_ptr = iseed;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (2*n) );

    LAPACK_claghe(
        &n_, &k_,
        D,
        (lapack_complex_float*) A, &lda_,
        iseed_ptr,
        (lapack_complex_float*) &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( iseed_.begin(), iseed_.end(), iseed );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t laghe(
    int64_t n, int64_t k,
    double const* D,
    std::complex<double>* A, int64_t lda,
    int64_t* iseed )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > iseed_( &iseed[0], &iseed[(4)] );
        blas_int* iseed_ptr = &iseed_[0];
    #else
        blas_int* iseed_ptr = iseed;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (2*n) );

    LAPACK_zlaghe(
        &n_, &k_,
        D,
        (lapack_complex_double*) A, &lda_,
        iseed_ptr,
        (lapack_complex_double*) &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( iseed_.begin(), iseed_.end(), iseed );
    #endif
    return info_;
}

}  // namespace lapack

#endif  // LAPACK_MATGEN
