#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t gelsd(
    int64_t m, int64_t n, int64_t nrhs,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    float* S, float rcond,
    int64_t* rank )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int rank_ = (blas_int) *rank;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int qry_iwork[1];
    blas_int ineg_one = -1;
    LAPACK_sgelsd( &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, S, &rcond, &rank_, qry_work, &ineg_one, qry_iwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );
    std::vector< blas_int > iwork( (max( (int64_t) 1, liwork)) );

    LAPACK_sgelsd( &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, S, &rcond, &rank_, &work[0], &lwork_, &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *rank = rank_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gelsd(
    int64_t m, int64_t n, int64_t nrhs,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    double* S, double rcond,
    int64_t* rank )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int rank_ = (blas_int) *rank;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int qry_iwork[1];
    blas_int ineg_one = -1;
    LAPACK_dgelsd( &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, S, &rcond, &rank_, qry_work, &ineg_one, qry_iwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );
    std::vector< blas_int > iwork( (max( (int64_t) 1, liwork)) );

    LAPACK_dgelsd( &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, S, &rcond, &rank_, &work[0], &lwork_, &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *rank = rank_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gelsd(
    int64_t m, int64_t n, int64_t nrhs,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    float* S, float rcond,
    int64_t* rank )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int rank_ = (blas_int) *rank;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    blas_int qry_iwork[1];
    blas_int ineg_one = -1;
    LAPACK_cgelsd( &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, S, &rcond, &rank_, qry_work, &ineg_one, qry_rwork, qry_iwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );
    std::vector< float > rwork( (max( (int64_t) 1, lrwork)) );
    std::vector< blas_int > iwork( (max( (int64_t) 1, liwork)) );

    LAPACK_cgelsd( &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, S, &rcond, &rank_, &work[0], &lwork_, &rwork[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *rank = rank_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gelsd(
    int64_t m, int64_t n, int64_t nrhs,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    double* S, double rcond,
    int64_t* rank )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int rank_ = (blas_int) *rank;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    blas_int qry_iwork[1];
    blas_int ineg_one = -1;
    LAPACK_zgelsd( &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, S, &rcond, &rank_, qry_work, &ineg_one, qry_rwork, qry_iwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );
    std::vector< double > rwork( (max( (int64_t) 1, lrwork)) );
    std::vector< blas_int > iwork( (max( (int64_t) 1, liwork)) );

    LAPACK_zgelsd( &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, S, &rcond, &rank_, &work[0], &lwork_, &rwork[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *rank = rank_;
    return info_;
}

}  // namespace lapack
