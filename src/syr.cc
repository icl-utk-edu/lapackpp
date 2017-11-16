#include "lapack.hh"
#include "lapack_fortran.h"

#if LAPACK_VERSION_MAJOR >= 3 && LAPACK_VERSION_MINOR >= 5  // >= 3.5

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
void syr(
    lapack::Uplo uplo, int64_t n, std::complex<float> alpha,
    std::complex<float> const* X, int64_t incx,
    std::complex<float>* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int lda_ = (blas_int) lda;

    LAPACK_csyr( &uplo_, &n_, &alpha, X, &incx_, A, &lda_ );
}

// -----------------------------------------------------------------------------
void syr(
    lapack::Uplo uplo, int64_t n, std::complex<double> alpha,
    std::complex<double> const* X, int64_t incx,
    std::complex<double>* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int lda_ = (blas_int) lda;

    LAPACK_zsyr( &uplo_, &n_, &alpha, X, &incx_, A, &lda_ );
}

}  // namespace lapack

#endif  // LAPACK >= 3.5
