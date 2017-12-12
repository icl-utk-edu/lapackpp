#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
float lanhb(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<float> const* AB, int64_t ldab )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kd) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) kd;
    blas_int ldab_ = (blas_int) ldab;

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    std::vector< float > work( max(1,lwork) );

    return LAPACK_clanhb( &norm_, &uplo_, &n_, &k_, AB, &ldab_, &work[0] );
}

// -----------------------------------------------------------------------------
double lanhb(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n, int64_t kd,
    std::complex<double> const* AB, int64_t ldab )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kd) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) kd;
    blas_int ldab_ = (blas_int) ldab;

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    std::vector< double > work( max(1,lwork) );

    return LAPACK_zlanhb( &norm_, &uplo_, &n_, &k_, AB, &ldab_, &work[0] );
}

}  // namespace lapack
