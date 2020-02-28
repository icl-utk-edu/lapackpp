#include "lapack.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t tbcon(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag, int64_t n, int64_t kd,
    float const* AB, int64_t ldab,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (3*n) );
    std::vector< lapack_int > iwork( (n) );

    LAPACK_stbcon(
        &norm_, &uplo_, &diag_, &n_, &kd_,
        AB, &ldab_, rcond,
        &work[0],
        &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tbcon(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag, int64_t n, int64_t kd,
    double const* AB, int64_t ldab,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (3*n) );
    std::vector< lapack_int > iwork( (n) );

    LAPACK_dtbcon(
        &norm_, &uplo_, &diag_, &n_, &kd_,
        AB, &ldab_, rcond,
        &work[0],
        &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tbcon(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag, int64_t n, int64_t kd,
    std::complex<float> const* AB, int64_t ldab,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (2*n) );
    std::vector< float > rwork( (n) );

    LAPACK_ctbcon(
        &norm_, &uplo_, &diag_, &n_, &kd_,
        (lapack_complex_float*) AB, &ldab_, rcond,
        (lapack_complex_float*) &work[0],
        &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tbcon(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag, int64_t n, int64_t kd,
    std::complex<double> const* AB, int64_t ldab,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (2*n) );
    std::vector< double > rwork( (n) );

    LAPACK_ztbcon(
        &norm_, &uplo_, &diag_, &n_, &kd_,
        (lapack_complex_double*) AB, &ldab_, rcond,
        (lapack_complex_double*) &work[0],
        &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
