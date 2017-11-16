#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
void lacgv(
    int64_t n,
    std::complex<float>* X, int64_t incx )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int incx_ = (blas_int) incx;

    LAPACK_clacgv( &n_, X, &incx_ );
}

// -----------------------------------------------------------------------------
void lacgv(
    int64_t n,
    std::complex<double>* X, int64_t incx )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int incx_ = (blas_int) incx;

    LAPACK_zlacgv( &n_, X, &incx_ );
}

}  // namespace lapack
