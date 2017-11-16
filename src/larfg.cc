#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
void larfg(
    int64_t n,
    float* alpha,
    float* X, int64_t incx,
    float* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int incx_ = (blas_int) incx;

    LAPACK_slarfg( &n_, alpha, X, &incx_, tau );
}

// -----------------------------------------------------------------------------
void larfg(
    int64_t n,
    double* alpha,
    double* X, int64_t incx,
    double* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int incx_ = (blas_int) incx;

    LAPACK_dlarfg( &n_, alpha, X, &incx_, tau );
}

// -----------------------------------------------------------------------------
void larfg(
    int64_t n,
    std::complex<float>* alpha,
    std::complex<float>* X, int64_t incx,
    std::complex<float>* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int incx_ = (blas_int) incx;

    LAPACK_clarfg( &n_, alpha, X, &incx_, tau );
}

// -----------------------------------------------------------------------------
void larfg(
    int64_t n,
    std::complex<double>* alpha,
    std::complex<double>* X, int64_t incx,
    std::complex<double>* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int incx_ = (blas_int) incx;

    LAPACK_zlarfg( &n_, alpha, X, &incx_, tau );
}

}  // namespace lapack
