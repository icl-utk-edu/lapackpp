#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t disna(
    lapack::Job job, int64_t m, int64_t n,
    float const* D,
    float* SEP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_sdisna( &job_, &m_, &n_, D, SEP, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t disna(
    lapack::Job job, int64_t m, int64_t n,
    double const* D,
    double* SEP )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    LAPACK_ddisna( &job_, &m_, &n_, D, SEP, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
