#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t sygvx(
    int64_t itype, lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n,
    float* A, int64_t lda,
    float* B, int64_t ldb, float vl, float vu, int64_t il, int64_t iu, float abstol,
    int64_t* m,
    float* W,
    float* Z, int64_t ldz,
    int64_t* ifail )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(itype) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(il) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(iu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int itype_ = (lapack_int) itype;
    char jobz_ = job2char( jobz );
    char range_ = range2char( range );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int il_ = (lapack_int) il;
    lapack_int iu_ = (lapack_int) iu;
    lapack_int m_ = (lapack_int) *m;
    lapack_int ldz_ = (lapack_int) ldz;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ifail_( (n) );
        lapack_int* ifail_ptr = &ifail_[0];
    #else
        lapack_int* ifail_ptr = ifail;
    #endif
    lapack_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_ssygvx(
        &itype_, &jobz_, &range_, &uplo_, &n_,
        A, &lda_,
        B, &ldb_, &vl, &vu, &il_, &iu_, &abstol, &m_,
        W,
        Z, &ldz_,
        qry_work, &ineg_one,
        qry_iwork,
        ifail_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );
    std::vector< lapack_int > iwork( (5*n) );

    LAPACK_ssygvx(
        &itype_, &jobz_, &range_, &uplo_, &n_,
        A, &lda_,
        B, &ldb_, &vl, &vu, &il_, &iu_, &abstol, &m_,
        W,
        Z, &ldz_,
        &work[0], &lwork_,
        &iwork[0],
        ifail_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    #ifndef LAPACK_ILP64
        std::copy( ifail_.begin(), ifail_.end(), ifail );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t sygvx(
    int64_t itype, lapack::Job jobz, lapack::Range range, lapack::Uplo uplo, int64_t n,
    double* A, int64_t lda,
    double* B, int64_t ldb, double vl, double vu, int64_t il, int64_t iu, double abstol,
    int64_t* m,
    double* W,
    double* Z, int64_t ldz,
    int64_t* ifail )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(itype) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(il) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(iu) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int itype_ = (lapack_int) itype;
    char jobz_ = job2char( jobz );
    char range_ = range2char( range );
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int il_ = (lapack_int) il;
    lapack_int iu_ = (lapack_int) iu;
    lapack_int m_ = (lapack_int) *m;
    lapack_int ldz_ = (lapack_int) ldz;
    #ifndef LAPACK_ILP64
        // 32-bit copy
        std::vector< lapack_int > ifail_( (n) );
        lapack_int* ifail_ptr = &ifail_[0];
    #else
        lapack_int* ifail_ptr = ifail;
    #endif
    lapack_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    lapack_int qry_iwork[1];
    lapack_int ineg_one = -1;
    LAPACK_dsygvx(
        &itype_, &jobz_, &range_, &uplo_, &n_,
        A, &lda_,
        B, &ldb_, &vl, &vu, &il_, &iu_, &abstol, &m_,
        W,
        Z, &ldz_,
        qry_work, &ineg_one,
        qry_iwork,
        ifail_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    lapack_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );
    std::vector< lapack_int > iwork( (5*n) );

    LAPACK_dsygvx(
        &itype_, &jobz_, &range_, &uplo_, &n_,
        A, &lda_,
        B, &ldb_, &vl, &vu, &il_, &iu_, &abstol, &m_,
        W,
        Z, &ldz_,
        &work[0], &lwork_,
        &iwork[0],
        ifail_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *m = m_;
    #ifndef LAPACK_ILP64
        std::copy( ifail_.begin(), ifail_.end(), ifail );
    #endif
    return info_;
}

}  // namespace lapack
