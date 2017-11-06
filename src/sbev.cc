#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t sbev(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t kd,
    float* AB, int64_t ldab,
    float* W,
    float* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kd) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int kd_ = (blas_int) kd;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (max( (int64_t) 1, 3*n-2)) );

    LAPACK_ssbev( &jobz_, &uplo_, &n_, &kd_, AB, &ldab_, W, Z, &ldz_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t sbev(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t kd,
    double* AB, int64_t ldab,
    double* W,
    double* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kd) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int kd_ = (blas_int) kd;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (max( (int64_t) 1, 3*n-2)) );

    LAPACK_dsbev( &jobz_, &uplo_, &n_, &kd_, AB, &ldab_, W, Z, &ldz_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
