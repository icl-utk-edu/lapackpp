#include "lapack.hh"
#include "lapack_fortran.h"

#ifdef LAPACK_MATGEN

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t lagge(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    float const* D,
    float* A, int64_t lda,
    int64_t* iseed )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
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
    std::vector< float > work( (m+n) );

    LAPACK_slagge( &m_, &n_, &kl_, &ku_, D, A, &lda_, iseed_ptr, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( iseed_.begin(), iseed_.end(), iseed );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t lagge(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    double const* D,
    double* A, int64_t lda,
    int64_t* iseed )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
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
    std::vector< double > work( (m+n) );

    LAPACK_dlagge( &m_, &n_, &kl_, &ku_, D, A, &lda_, iseed_ptr, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( iseed_.begin(), iseed_.end(), iseed );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t lagge(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    float const* D,
    std::complex<float>* A, int64_t lda,
    int64_t* iseed )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
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
    std::vector< std::complex<float> > work( (m+n) );

    LAPACK_clagge( &m_, &n_, &kl_, &ku_, D, A, &lda_, iseed_ptr, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( iseed_.begin(), iseed_.end(), iseed );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t lagge(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    double const* D,
    std::complex<double>* A, int64_t lda,
    int64_t* iseed )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
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
    std::vector< std::complex<double> > work( (m+n) );

    LAPACK_zlagge( &m_, &n_, &kl_, &ku_, D, A, &lda_, iseed_ptr, &work[0], &info_ );
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
