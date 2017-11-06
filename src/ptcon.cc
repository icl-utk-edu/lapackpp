#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t ptcon(
    int64_t n,
    float const* D,
    float const* E, float anorm,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (n) );

    LAPACK_sptcon( &n_, D, E, &anorm, rcond, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ptcon(
    int64_t n,
    double const* D,
    double const* E, double anorm,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (n) );

    LAPACK_dptcon( &n_, D, E, &anorm, rcond, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ptcon(
    int64_t n,
    float const* D,
    std::complex<float> const* E, float anorm,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > rwork( (n) );

    LAPACK_cptcon( &n_, D, E, &anorm, rcond, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ptcon(
    int64_t n,
    double const* D,
    std::complex<double> const* E, double anorm,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > rwork( (n) );

    LAPACK_zptcon( &n_, D, E, &anorm, rcond, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
