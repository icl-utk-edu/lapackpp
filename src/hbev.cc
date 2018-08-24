#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t hbev(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<float>* AB, int64_t ldab,
    float* W,
    std::complex<float>* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int ldz_ = (lapack_int) ldz;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (n) );
    std::vector< float > rwork( (max( 1, 3*n-2 )) );

    LAPACK_chbev(
        &jobz_, &uplo_, &n_, &kd_,
        (lapack_complex_float*) AB, &ldab_,
        W,
        (lapack_complex_float*) Z, &ldz_,
        (lapack_complex_float*) &work[0],
        &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t hbev(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<double>* AB, int64_t ldab,
    double* W,
    std::complex<double>* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int kd_ = (lapack_int) kd;
    lapack_int ldab_ = (lapack_int) ldab;
    lapack_int ldz_ = (lapack_int) ldz;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (n) );
    std::vector< double > rwork( (max( 1, 3*n-2 )) );

    LAPACK_zhbev(
        &jobz_, &uplo_, &n_, &kd_,
        (lapack_complex_double*) AB, &ldab_,
        W,
        (lapack_complex_double*) Z, &ldz_,
        (lapack_complex_double*) &work[0],
        &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
