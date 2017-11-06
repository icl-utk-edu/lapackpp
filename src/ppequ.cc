#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t ppequ(
    lapack::Uplo uplo, int64_t n,
    float const* AP,
    float* S,
    float* scond,
    float* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_sppequ( &uplo_, &n_, AP, S, scond, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ppequ(
    lapack::Uplo uplo, int64_t n,
    double const* AP,
    double* S,
    double* scond,
    double* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_dppequ( &uplo_, &n_, AP, S, scond, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ppequ(
    lapack::Uplo uplo, int64_t n,
    std::complex<float> const* AP,
    float* S,
    float* scond,
    float* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_cppequ( &uplo_, &n_, AP, S, scond, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ppequ(
    lapack::Uplo uplo, int64_t n,
    std::complex<double> const* AP,
    double* S,
    double* scond,
    double* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_zppequ( &uplo_, &n_, AP, S, scond, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
