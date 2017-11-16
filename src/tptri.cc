#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t tptri(
    lapack::Uplo uplo, lapack::Diag diag, int64_t n,
    float* AP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_stptri( &uplo_, &diag_, &n_, AP, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tptri(
    lapack::Uplo uplo, lapack::Diag diag, int64_t n,
    double* AP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_dtptri( &uplo_, &diag_, &n_, AP, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tptri(
    lapack::Uplo uplo, lapack::Diag diag, int64_t n,
    std::complex<float>* AP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_ctptri( &uplo_, &diag_, &n_, AP, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tptri(
    lapack::Uplo uplo, lapack::Diag diag, int64_t n,
    std::complex<double>* AP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_ztptri( &uplo_, &diag_, &n_, AP, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
