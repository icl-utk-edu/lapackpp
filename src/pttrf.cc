#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t pttrf(
    int64_t n,
    float* D,
    float* E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_spttrf( &n_, D, E, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t pttrf(
    int64_t n,
    double* D,
    double* E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_dpttrf( &n_, D, E, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t pttrf(
    int64_t n,
    float* D,
    std::complex<float>* E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_cpttrf( &n_, D, E, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t pttrf(
    int64_t n,
    double* D,
    std::complex<double>* E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_zpttrf( &n_, D, E, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
