#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t sterf(
    int64_t n,
    float* D,
    float* E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    LAPACK_ssterf(
        &n_,
        D,
        E, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t sterf(
    int64_t n,
    double* D,
    double* E )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int n_ = (lapack_int) n;
    lapack_int info_ = 0;

    LAPACK_dsterf(
        &n_,
        D,
        E, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
