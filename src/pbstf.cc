#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t pbstf(
    lapack::Uplo uplo, int64_t n, int64_t kd,
    float* AB, int64_t ldab )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kd) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int kd_ = (blas_int) kd;
    blas_int ldab_ = (blas_int) ldab;
    blas_int info_ = 0;

    LAPACK_spbstf( &uplo_, &n_, &kd_, AB, &ldab_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t pbstf(
    lapack::Uplo uplo, int64_t n, int64_t kd,
    double* AB, int64_t ldab )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kd) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int kd_ = (blas_int) kd;
    blas_int ldab_ = (blas_int) ldab;
    blas_int info_ = 0;

    LAPACK_dpbstf( &uplo_, &n_, &kd_, AB, &ldab_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t pbstf(
    lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<float>* AB, int64_t ldab )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kd) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int kd_ = (blas_int) kd;
    blas_int ldab_ = (blas_int) ldab;
    blas_int info_ = 0;

    LAPACK_cpbstf( &uplo_, &n_, &kd_, AB, &ldab_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t pbstf(
    lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<double>* AB, int64_t ldab )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kd) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int kd_ = (blas_int) kd;
    blas_int ldab_ = (blas_int) ldab;
    blas_int info_ = 0;

    LAPACK_zpbstf( &uplo_, &n_, &kd_, AB, &ldab_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
