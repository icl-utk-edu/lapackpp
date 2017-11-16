#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t hptrd(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* AP,
    float* D,
    float* E,
    std::complex<float>* TAU )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_chptrd( &uplo_, &n_, AP, D, E, TAU, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t hptrd(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* AP,
    double* D,
    double* E,
    std::complex<double>* TAU )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_zhptrd( &uplo_, &n_, AP, D, E, TAU, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
