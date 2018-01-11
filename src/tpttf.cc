#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t tpttf(
    lapack::Op transr, lapack::Uplo uplo, int64_t n,
    float const* AP,
    float* ARF )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char transr_ = op2char( transr );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_stpttf( &transr_, &uplo_, &n_, AP, ARF, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tpttf(
    lapack::Op transr, lapack::Uplo uplo, int64_t n,
    double const* AP,
    double* ARF )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char transr_ = op2char( transr );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_dtpttf( &transr_, &uplo_, &n_, AP, ARF, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tpttf(
    lapack::Op transr, lapack::Uplo uplo, int64_t n,
    std::complex<float> const* AP,
    std::complex<float>* ARF )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char transr_ = op2char( transr );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_ctpttf( &transr_, &uplo_, &n_, AP, ARF, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tpttf(
    lapack::Op transr, lapack::Uplo uplo, int64_t n,
    std::complex<double> const* AP,
    std::complex<double>* ARF )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char transr_ = op2char( transr );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_ztpttf( &transr_, &uplo_, &n_, AP, ARF, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
