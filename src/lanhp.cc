#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
float lanhp(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    std::complex<float> const* AP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    std::vector< float > work( max(1,lwork) );

    return LAPACK_clanhp( &norm_, &uplo_, &n_, AP, &work[0] );
}

// -----------------------------------------------------------------------------
double lanhp(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    std::complex<double> const* AP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;

    // from docs
    int64_t lwork = (norm == Norm::Inf || norm == Norm::One ? n : 1);

    // allocate workspace
    std::vector< double > work( max(1,lwork) );

    return LAPACK_zlanhp( &norm_, &uplo_, &n_, AP, &work[0] );
}

}  // namespace lapack
