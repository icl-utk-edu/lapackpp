#include "lapack.hh"
#include "lapack_fortran.h"

#if LAPACK_VERSION_MAJOR >= 3 && LAPACK_VERSION_MINOR >= 6  // >= v3.6

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t bdsvdx(
    lapack::Uplo uplo, lapack::Job jobz, lapack::Range range, int64_t n,
    float* D,
    float* E, float vl, float vu, int64_t il, int64_t iu,
    int64_t* ns,
    float* S,
    float* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(il) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(iu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char jobz_ = job2char( jobz );
    char range_ = range2char( range );
    blas_int n_ = (blas_int) n;
    blas_int il_ = (blas_int) il;
    blas_int iu_ = (blas_int) iu;
    blas_int ns_ = (blas_int) *ns;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (14*n) );
    std::vector< blas_int > iwork( (12*n) );

    LAPACK_sbdsvdx( &uplo_, &jobz_, &range_, &n_, D, E, &vl, &vu, &il_, &iu_, &ns_, S, Z, &ldz_, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ns = ns_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t bdsvdx(
    lapack::Uplo uplo, lapack::Job jobz, lapack::Range range, int64_t n,
    double* D,
    double* E, double vl, double vu, int64_t il, int64_t iu,
    int64_t* ns,
    double* S,
    double* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(il) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(iu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    char jobz_ = job2char( jobz );
    char range_ = range2char( range );
    blas_int n_ = (blas_int) n;
    blas_int il_ = (blas_int) il;
    blas_int iu_ = (blas_int) iu;
    blas_int ns_ = (blas_int) *ns;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (14*n) );
    std::vector< blas_int > iwork( (12*n) );

    LAPACK_dbdsvdx( &uplo_, &jobz_, &range_, &n_, D, E, &vl, &vu, &il_, &iu_, &ns_, S, Z, &ldz_, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ns = ns_;
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= v3.6
