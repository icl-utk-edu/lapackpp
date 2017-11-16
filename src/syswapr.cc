#include "lapack.hh"
#include "lapack_fortran.h"

#if LAPACK_VERSION_MAJOR >= 3 && LAPACK_VERSION_MINOR >= 3  // >= 3.3

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
void syswapr(
    lapack::Uplo uplo, int64_t n,
    float* A, int64_t lda, int64_t i1, int64_t i2 )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(i1) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(i2) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int i1_ = (blas_int) i1;
    blas_int i2_ = (blas_int) i2;

    LAPACK_ssyswapr( &uplo_, &n_, A, &lda_, &i1_, &i2_ );
}

// -----------------------------------------------------------------------------
void syswapr(
    lapack::Uplo uplo, int64_t n,
    double* A, int64_t lda, int64_t i1, int64_t i2 )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(i1) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(i2) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int i1_ = (blas_int) i1;
    blas_int i2_ = (blas_int) i2;

    LAPACK_dsyswapr( &uplo_, &n_, A, &lda_, &i1_, &i2_ );
}

// -----------------------------------------------------------------------------
void syswapr(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda, int64_t i1, int64_t i2 )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(i1) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(i2) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int i1_ = (blas_int) i1;
    blas_int i2_ = (blas_int) i2;

    LAPACK_csyswapr( &uplo_, &n_, A, &lda_, &i1_, &i2_ );
}

// -----------------------------------------------------------------------------
void syswapr(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda, int64_t i1, int64_t i2 )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(i1) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(i2) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int i1_ = (blas_int) i1;
    blas_int i2_ = (blas_int) i2;

    LAPACK_zsyswapr( &uplo_, &n_, A, &lda_, &i1_, &i2_ );
}

}  // namespace lapack

#endif  // LAPACK >= 3.3.0
