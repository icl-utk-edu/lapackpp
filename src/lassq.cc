#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
void lassq(
    int64_t n,
    float const* X, int64_t incx,
    float* scale,
    float* sumsq )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int incx_ = (blas_int) incx;

    LAPACK_slassq( &n_, X, &incx_, scale, sumsq );
}

// -----------------------------------------------------------------------------
void lassq(
    int64_t n,
    double const* X, int64_t incx,
    double* scale,
    double* sumsq )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int incx_ = (blas_int) incx;

    LAPACK_dlassq( &n_, X, &incx_, scale, sumsq );
}

// -----------------------------------------------------------------------------
void lassq(
    int64_t n,
    std::complex<float> const* X, int64_t incx,
    float* scale,
    float* sumsq )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int incx_ = (blas_int) incx;

    LAPACK_classq( &n_, X, &incx_, scale, sumsq );
}

// -----------------------------------------------------------------------------
void lassq(
    int64_t n,
    std::complex<double> const* X, int64_t incx,
    double* scale,
    double* sumsq )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int incx_ = (blas_int) incx;

    LAPACK_zlassq( &n_, X, &incx_, scale, sumsq );
}

}  // namespace lapack
