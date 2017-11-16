#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t gelsy(
    int64_t m, int64_t n, int64_t nrhs,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    int64_t* jpvt, float rcond,
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
    #if 1
        // 32-bit copy
        std::vector< blas_int > jpvt_( &jpvt[0], &jpvt[(n)] );
        blas_int* jpvt_ptr = &jpvt_[0];
    #else
        blas_int* jpvt_ptr = jpvt;
    #endif
    blas_int rank_ = (blas_int) *rank;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_sgelsy( &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, jpvt_ptr, &rcond, &rank_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_sgelsy( &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, jpvt_ptr, &rcond, &rank_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( jpvt_.begin(), jpvt_.end(), jpvt );
    #endif
    *rank = rank_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gelsy(
    int64_t m, int64_t n, int64_t nrhs,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    int64_t* jpvt, double rcond,
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
    #if 1
        // 32-bit copy
        std::vector< blas_int > jpvt_( &jpvt[0], &jpvt[(n)] );
        blas_int* jpvt_ptr = &jpvt_[0];
    #else
        blas_int* jpvt_ptr = jpvt;
    #endif
    blas_int rank_ = (blas_int) *rank;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dgelsy( &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, jpvt_ptr, &rcond, &rank_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dgelsy( &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, jpvt_ptr, &rcond, &rank_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( jpvt_.begin(), jpvt_.end(), jpvt );
    #endif
    *rank = rank_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gelsy(
    int64_t m, int64_t n, int64_t nrhs,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    int64_t* jpvt, float rcond,
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
    #if 1
        // 32-bit copy
        std::vector< blas_int > jpvt_( &jpvt[0], &jpvt[(n)] );
        blas_int* jpvt_ptr = &jpvt_[0];
    #else
        blas_int* jpvt_ptr = jpvt;
    #endif
    blas_int rank_ = (blas_int) *rank;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_cgelsy( &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, jpvt_ptr, &rcond, &rank_, qry_work, &ineg_one, qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );
    std::vector< float > rwork( (2*n) );

    LAPACK_cgelsy( &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, jpvt_ptr, &rcond, &rank_, &work[0], &lwork_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( jpvt_.begin(), jpvt_.end(), jpvt );
    #endif
    *rank = rank_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gelsy(
    int64_t m, int64_t n, int64_t nrhs,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    int64_t* jpvt, double rcond,
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
    #if 1
        // 32-bit copy
        std::vector< blas_int > jpvt_( &jpvt[0], &jpvt[(n)] );
        blas_int* jpvt_ptr = &jpvt_[0];
    #else
        blas_int* jpvt_ptr = jpvt;
    #endif
    blas_int rank_ = (blas_int) *rank;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_zgelsy( &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, jpvt_ptr, &rcond, &rank_, qry_work, &ineg_one, qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );
    std::vector< double > rwork( (2*n) );

    LAPACK_zgelsy( &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, jpvt_ptr, &rcond, &rank_, &work[0], &lwork_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( jpvt_.begin(), jpvt_.end(), jpvt );
    #endif
    *rank = rank_;
    return info_;
}

}  // namespace lapack
