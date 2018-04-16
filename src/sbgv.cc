#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t sbgv(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t ka, int64_t kb,
    float* AB, int64_t ldab,
    float* BB, int64_t ldbb,
    float* W,
    float* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ka) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(kb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldbb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int ka_ = (blas_int) ka;
    blas_int kb_ = (blas_int) kb;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldbb_ = (blas_int) ldbb;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (3*n) );

    LAPACK_ssbgv(
        &jobz_, &uplo_, &n_, &ka_, &kb_,
        AB, &ldab_,
        BB, &ldbb_,
        W,
        Z, &ldz_,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t sbgv(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t ka, int64_t kb,
    double* AB, int64_t ldab,
    double* BB, int64_t ldbb,
    double* W,
    double* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ka) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(kb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldbb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int ka_ = (blas_int) ka;
    blas_int kb_ = (blas_int) kb;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldbb_ = (blas_int) ldbb;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (3*n) );

    LAPACK_dsbgv(
        &jobz_, &uplo_, &n_, &ka_, &kb_,
        AB, &ldab_,
        BB, &ldbb_,
        W,
        Z, &ldz_,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
