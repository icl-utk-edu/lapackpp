#include "lapack.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t tptri(
    lapack::Uplo uplo, lapack::Diag diag, int64_t n,
    float* AP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    LAPACK_stptri(
        &uplo_, &diag_, &n_,
        AP, &info_ );
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
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    LAPACK_dtptri(
        &uplo_, &diag_, &n_,
        AP, &info_ );
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
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    LAPACK_ctptri(
        &uplo_, &diag_, &n_,
        (lapack_complex_float*) AP, &info_ );
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
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    LAPACK_ztptri(
        &uplo_, &diag_, &n_,
        (lapack_complex_double*) AP, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
