#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t disna(
    lapack::JobCond jobcond, int64_t m, int64_t n,
    float const* D,
    float* SEP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char jobcond_ = jobcond2char( jobcond );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_sdisna( &jobcond_, &m_, &n_, D, SEP, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t disna(
    lapack::JobCond jobcond, int64_t m, int64_t n,
    double const* D,
    double* SEP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char jobcond_ = jobcond2char( jobcond );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_ddisna( &jobcond_, &m_, &n_, D, SEP, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
