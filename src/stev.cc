#include "lapack.hh"
#include "lapack/fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t stev(
    lapack::Job jobz, int64_t n,
    float* D,
    float* E,
    float* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    char jobz_ = job2char( jobz );
    lapack_int n_ = (lapack_int) n;
    lapack_int ldz_ = (lapack_int) ldz;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (max( 1, 2*n-2 )) );

    LAPACK_sstev(
        &jobz_, &n_,
        D,
        E,
        Z, &ldz_,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t stev(
    lapack::Job jobz, int64_t n,
    double* D,
    double* E,
    double* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    char jobz_ = job2char( jobz );
    lapack_int n_ = (lapack_int) n;
    lapack_int ldz_ = (lapack_int) ldz;
    lapack_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (max( 1, 2*n-2 )) );

    LAPACK_dstev(
        &jobz_, &n_,
        D,
        E,
        Z, &ldz_,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
