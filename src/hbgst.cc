#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t hbgst(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t ka, int64_t kb,
    std::complex<float>* AB, int64_t ldab,
    std::complex<float> const* BB, int64_t ldbb,
    std::complex<float>* X, int64_t ldx )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ka) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(kb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldbb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int ka_ = (blas_int) ka;
    blas_int kb_ = (blas_int) kb;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldbb_ = (blas_int) ldbb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (n) );
    std::vector< float > rwork( (n) );

    LAPACK_chbgst( &jobz_, &uplo_, &n_, &ka_, &kb_, AB, &ldab_, BB, &ldbb_, X, &ldx_, &work[0], &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t hbgst(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t ka, int64_t kb,
    std::complex<double>* AB, int64_t ldab,
    std::complex<double> const* BB, int64_t ldbb,
    std::complex<double>* X, int64_t ldx )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ka) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(kb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldbb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
    }
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int ka_ = (blas_int) ka;
    blas_int kb_ = (blas_int) kb;
    blas_int ldab_ = (blas_int) ldab;
    blas_int ldbb_ = (blas_int) ldbb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (n) );
    std::vector< double > rwork( (n) );

    LAPACK_zhbgst( &jobz_, &uplo_, &n_, &ka_, &kb_, AB, &ldab_, BB, &ldbb_, X, &ldx_, &work[0], &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
