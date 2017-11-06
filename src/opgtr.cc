#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t opgtr(
    lapack::Uplo uplo, int64_t n,
    float const* AP,
    float const* TAU,
    float* Q, int64_t ldq )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int ldq_ = (blas_int) ldq;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (n-1) );

    LAPACK_sopgtr( &uplo_, &n_, AP, TAU, Q, &ldq_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t opgtr(
    lapack::Uplo uplo, int64_t n,
    double const* AP,
    double const* TAU,
    double* Q, int64_t ldq )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int ldq_ = (blas_int) ldq;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (n-1) );

    LAPACK_dopgtr( &uplo_, &n_, AP, TAU, Q, &ldq_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
