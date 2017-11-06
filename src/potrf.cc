#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t potrf(
    lapack::Uplo uplo, int64_t n,
    float* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    LAPACK_spotrf( &uplo_, &n_, A, &lda_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t potrf(
    lapack::Uplo uplo, int64_t n,
    double* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    LAPACK_dpotrf( &uplo_, &n_, A, &lda_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t potrf(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    LAPACK_cpotrf( &uplo_, &n_, A, &lda_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t potrf(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    LAPACK_zpotrf( &uplo_, &n_, A, &lda_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
