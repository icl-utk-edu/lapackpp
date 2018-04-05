#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t sbtrd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t kd,
    float* AB, int64_t ldab,
    float* D,
    float* E,
    float* Q, int64_t ldq )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int kd_ = (blas_int) kd;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldq_ = (blas_int) ldq;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (n) );

    LAPACK_ssbtrd( &jobz_, &uplo_, &n_, &kd_, AB, &ldab_, D, E, Q, &ldq_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t sbtrd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t kd,
    double* AB, int64_t ldab,
    double* D,
    double* E,
    double* Q, int64_t ldq )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(kd) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int kd_ = (blas_int) kd;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldq_ = (blas_int) ldq;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (n) );

    LAPACK_dsbtrd( &jobz_, &uplo_, &n_, &kd_, AB, &ldab_, D, E, Q, &ldq_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
