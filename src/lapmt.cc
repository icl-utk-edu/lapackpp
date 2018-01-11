#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
void lapmt(
    bool forwrd, int64_t m, int64_t n,
    float* X, int64_t ldx,
    int64_t* K )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(forwrd) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int forwrd_ = (blas_int) forwrd;
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ldx_ = (blas_int) ldx;
    #if 1
        // 32-bit copy
        std::vector< blas_int > K_( &K[0], &K[(n)] );
        blas_int* K_ptr = &K_[0];
    #else
        blas_int* K_ptr = K;
    #endif

    LAPACK_slapmt( &forwrd_, &m_, &n_, X, &ldx_, K_ptr );
    #if 1
        std::copy( K_.begin(), K_.end(), K );
    #endif
}

// -----------------------------------------------------------------------------
void lapmt(
    bool forwrd, int64_t m, int64_t n,
    double* X, int64_t ldx,
    int64_t* K )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(forwrd) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int forwrd_ = (blas_int) forwrd;
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ldx_ = (blas_int) ldx;
    #if 1
        // 32-bit copy
        std::vector< blas_int > K_( &K[0], &K[(n)] );
        blas_int* K_ptr = &K_[0];
    #else
        blas_int* K_ptr = K;
    #endif

    LAPACK_dlapmt( &forwrd_, &m_, &n_, X, &ldx_, K_ptr );
    #if 1
        std::copy( K_.begin(), K_.end(), K );
    #endif
}

// -----------------------------------------------------------------------------
void lapmt(
    bool forwrd, int64_t m, int64_t n,
    std::complex<float>* X, int64_t ldx,
    int64_t* K )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(forwrd) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int forwrd_ = (blas_int) forwrd;
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ldx_ = (blas_int) ldx;
    #if 1
        // 32-bit copy
        std::vector< blas_int > K_( &K[0], &K[(n)] );
        blas_int* K_ptr = &K_[0];
    #else
        blas_int* K_ptr = K;
    #endif

    LAPACK_clapmt( &forwrd_, &m_, &n_, X, &ldx_, K_ptr );
    #if 1
        std::copy( K_.begin(), K_.end(), K );
    #endif
}

// -----------------------------------------------------------------------------
void lapmt(
    bool forwrd, int64_t m, int64_t n,
    std::complex<double>* X, int64_t ldx,
    int64_t* K )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(forwrd) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int forwrd_ = (blas_int) forwrd;
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ldx_ = (blas_int) ldx;
    #if 1
        // 32-bit copy
        std::vector< blas_int > K_( &K[0], &K[(n)] );
        blas_int* K_ptr = &K_[0];
    #else
        blas_int* K_ptr = K;
    #endif

    LAPACK_zlapmt( &forwrd_, &m_, &n_, X, &ldx_, K_ptr );
    #if 1
        std::copy( K_.begin(), K_.end(), K );
    #endif
}

}  // namespace lapack
