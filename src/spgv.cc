#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t spgv(
    int64_t itype, lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    float* AP,
    float* BP,
    float* W,
    float* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(itype) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    blas_int itype_ = (blas_int) itype;
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (3*n) );

    LAPACK_sspgv( &itype_, &jobz_, &uplo_, &n_, AP, BP, W, Z, &ldz_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t spgv(
    int64_t itype, lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    double* AP,
    double* BP,
    double* W,
    double* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(itype) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    blas_int itype_ = (blas_int) itype;
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (3*n) );

    LAPACK_dspgv( &itype_, &jobz_, &uplo_, &n_, AP, BP, W, Z, &ldz_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
