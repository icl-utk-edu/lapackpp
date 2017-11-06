#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t sbgvd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t ka, int64_t kb,
    float* AB, int64_t ldab,
    float* BB, int64_t ldbb,
    float* W,
    float* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ka) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldbb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
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

    // query for workspace size
    float qry_work[1];
    blas_int qry_iwork[1];
    blas_int ineg_one = -1;
    LAPACK_ssbgvd( &jobz_, &uplo_, &n_, &ka_, &kb_, AB, &ldab_, BB, &ldbb_, W, Z, &ldz_, qry_work, &ineg_one, qry_iwork, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);
    blas_int liwork_ = real(qry_iwork[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );
    std::vector< blas_int > iwork( liwork_ );

    LAPACK_ssbgvd( &jobz_, &uplo_, &n_, &ka_, &kb_, AB, &ldab_, BB, &ldbb_, W, Z, &ldz_, &work[0], &lwork_, &iwork[0], &liwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t sbgvd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n, int64_t ka, int64_t kb,
    double* AB, int64_t ldab,
    double* BB, int64_t ldbb,
    double* W,
    double* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ka) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldbb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
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

    // query for workspace size
    double qry_work[1];
    blas_int qry_iwork[1];
    blas_int ineg_one = -1;
    LAPACK_dsbgvd( &jobz_, &uplo_, &n_, &ka_, &kb_, AB, &ldab_, BB, &ldbb_, W, Z, &ldz_, qry_work, &ineg_one, qry_iwork, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);
    blas_int liwork_ = real(qry_iwork[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );
    std::vector< blas_int > iwork( liwork_ );

    LAPACK_dsbgvd( &jobz_, &uplo_, &n_, &ka_, &kb_, AB, &ldab_, BB, &ldbb_, W, Z, &ldz_, &work[0], &lwork_, &iwork[0], &liwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
