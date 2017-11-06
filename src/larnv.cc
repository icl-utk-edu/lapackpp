#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
void larnv(
    int64_t idist,
    int64_t* iseed, int64_t n,
    float* X )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(idist) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    blas_int idist_ = (blas_int) idist;
    #if 1
        // 32-bit copy
        std::vector< blas_int > iseed_( &iseed[0], &iseed[(4)] );
        blas_int* iseed_ptr = &iseed_[0];
    #else
        blas_int* iseed_ptr = iseed;
    #endif
    blas_int n_ = (blas_int) n;

    LAPACK_slarnv( &idist_, iseed_ptr, &n_, X );
    #if 1
        std::copy( iseed_.begin(), iseed_.end(), iseed );
    #endif
}

// -----------------------------------------------------------------------------
void larnv(
    int64_t idist,
    int64_t* iseed, int64_t n,
    double* X )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(idist) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    blas_int idist_ = (blas_int) idist;
    #if 1
        // 32-bit copy
        std::vector< blas_int > iseed_( &iseed[0], &iseed[(4)] );
        blas_int* iseed_ptr = &iseed_[0];
    #else
        blas_int* iseed_ptr = iseed;
    #endif
    blas_int n_ = (blas_int) n;

    LAPACK_dlarnv( &idist_, iseed_ptr, &n_, X );
    #if 1
        std::copy( iseed_.begin(), iseed_.end(), iseed );
    #endif
}

// -----------------------------------------------------------------------------
void larnv(
    int64_t idist,
    int64_t* iseed, int64_t n,
    std::complex<float>* X )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(idist) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    blas_int idist_ = (blas_int) idist;
    #if 1
        // 32-bit copy
        std::vector< blas_int > iseed_( &iseed[0], &iseed[(4)] );
        blas_int* iseed_ptr = &iseed_[0];
    #else
        blas_int* iseed_ptr = iseed;
    #endif
    blas_int n_ = (blas_int) n;

    LAPACK_clarnv( &idist_, iseed_ptr, &n_, X );
    #if 1
        std::copy( iseed_.begin(), iseed_.end(), iseed );
    #endif
}

// -----------------------------------------------------------------------------
void larnv(
    int64_t idist,
    int64_t* iseed, int64_t n,
    std::complex<double>* X )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(idist) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    blas_int idist_ = (blas_int) idist;
    #if 1
        // 32-bit copy
        std::vector< blas_int > iseed_( &iseed[0], &iseed[(4)] );
        blas_int* iseed_ptr = &iseed_[0];
    #else
        blas_int* iseed_ptr = iseed;
    #endif
    blas_int n_ = (blas_int) n;

    LAPACK_zlarnv( &idist_, iseed_ptr, &n_, X );
    #if 1
        std::copy( iseed_.begin(), iseed_.end(), iseed );
    #endif
}

}  // namespace lapack
