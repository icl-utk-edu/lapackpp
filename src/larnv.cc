#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup initialize
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
/// @ingroup initialize
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
/// @ingroup initialize
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
/// Returns a vector of n random complex numbers from a uniform or
/// normal distribution.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] idist
///     The distribution of the random numbers:
///     - 1: real and imaginary parts each uniform (0,1)
///     - 2: real and imaginary parts each uniform (-1,1)
///     - 3: real and imaginary parts each normal (0,1)
///     - 4: uniformly distributed on the disc abs(z) < 1
///     - 5: uniformly distributed on the circle abs(z) = 1
///
/// @param[in,out] iseed
///     The vector iseed of length 4.
///     On entry, the seed of the random number generator; the array
///     elements must be between 0 and 4095, and iseed(4) must be
///     odd.
///     On exit, the seed is updated.
///
/// @param[in] n
///     The number of random numbers to be generated.
///
/// @param[out] X
///     The vector X of length n.
///     The generated random numbers.
///
// -----------------------------------------------------------------------------
/// @par Further Details
///
/// This routine calls the auxiliary routine `lapack::laruv` to generate random
/// real numbers from a uniform (0,1) distribution, in batches of up to
/// 128 using vectorisable code. The Box-Muller method is used to
/// transform numbers from a uniform to a normal distribution.
///
/// @ingroup initialize
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
