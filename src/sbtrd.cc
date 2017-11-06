#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t sbtrd(
    lapack::Vect vect, lapack::Uplo uplo, int64_t n, int64_t kd,
    float* AB, int64_t ldab,
    float* D,
    float* E,
    float* Q, int64_t ldq )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kd) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char vect_ = vect2char( vect );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int kd_ = (blas_int) kd;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldq_ = (blas_int) ldq;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (n) );

    LAPACK_ssbtrd( &vect_, &uplo_, &n_, &kd_, AB, &ldab_, D, E, Q, &ldq_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t sbtrd(
    lapack::Vect vect, lapack::Uplo uplo, int64_t n, int64_t kd,
    double* AB, int64_t ldab,
    double* D,
    double* E,
    double* Q, int64_t ldq )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kd) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char vect_ = vect2char( vect );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int kd_ = (blas_int) kd;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldq_ = (blas_int) ldq;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (n) );

    LAPACK_dsbtrd( &vect_, &uplo_, &n_, &kd_, AB, &ldab_, D, E, Q, &ldq_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
