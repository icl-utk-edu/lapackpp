#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t upgtr(
    lapack::Uplo uplo, int64_t n,
    std::complex<float> const* AP,
    std::complex<float> const* TAU,
    std::complex<float>* Q, int64_t ldq )
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
    std::vector< std::complex<float> > work( (n-1) );

    LAPACK_cupgtr( &uplo_, &n_, AP, TAU, Q, &ldq_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t upgtr(
    lapack::Uplo uplo, int64_t n,
    std::complex<double> const* AP,
    std::complex<double> const* TAU,
    std::complex<double>* Q, int64_t ldq )
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
    std::vector< std::complex<double> > work( (n-1) );

    LAPACK_zupgtr( &uplo_, &n_, AP, TAU, Q, &ldq_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
