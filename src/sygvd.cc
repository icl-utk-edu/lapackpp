#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t sygvd(
    int64_t itype, lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    float* W )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(itype) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int itype_ = (blas_int) itype;
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int qry_iwork[1];
    blas_int ineg_one = -1;
    LAPACK_ssygvd(
        &itype_, &jobz_, &uplo_, &n_,
        A, &lda_,
        B, &ldb_,
        W,
        qry_work, &ineg_one,
        qry_iwork, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);
    blas_int liwork_ = real(qry_iwork[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );
    std::vector< blas_int > iwork( liwork_ );

    LAPACK_ssygvd(
        &itype_, &jobz_, &uplo_, &n_,
        A, &lda_,
        B, &ldb_,
        W,
        &work[0], &lwork_,
        &iwork[0], &liwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t sygvd(
    int64_t itype, lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    double* W )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(itype) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int itype_ = (blas_int) itype;
    char jobz_ = job2char( jobz );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int qry_iwork[1];
    blas_int ineg_one = -1;
    LAPACK_dsygvd(
        &itype_, &jobz_, &uplo_, &n_,
        A, &lda_,
        B, &ldb_,
        W,
        qry_work, &ineg_one,
        qry_iwork, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);
    blas_int liwork_ = real(qry_iwork[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );
    std::vector< blas_int > iwork( liwork_ );

    LAPACK_dsygvd(
        &itype_, &jobz_, &uplo_, &n_,
        A, &lda_,
        B, &ldb_,
        W,
        &work[0], &lwork_,
        &iwork[0], &liwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
