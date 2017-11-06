#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t ggev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    float* ALPHAR,
    float* ALPHAI,
    float* BETA,
    float* VL, int64_t ldvl,
    float* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvr) > std::numeric_limits<blas_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldvl_ = (blas_int) ldvl;
    blas_int ldvr_ = (blas_int) ldvr;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_sggev( &jobvl_, &jobvr_, &n_, A, &lda_, B, &ldb_, ALPHAR, ALPHAI, BETA, VL, &ldvl_, VR, &ldvr_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_sggev( &jobvl_, &jobvr_, &n_, A, &lda_, B, &ldb_, ALPHAR, ALPHAI, BETA, VL, &ldvl_, VR, &ldvr_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    double* ALPHAR,
    double* ALPHAI,
    double* BETA,
    double* VL, int64_t ldvl,
    double* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvr) > std::numeric_limits<blas_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldvl_ = (blas_int) ldvl;
    blas_int ldvr_ = (blas_int) ldvr;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dggev( &jobvl_, &jobvr_, &n_, A, &lda_, B, &ldb_, ALPHAR, ALPHAI, BETA, VL, &ldvl_, VR, &ldvr_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dggev( &jobvl_, &jobvr_, &n_, A, &lda_, B, &ldb_, ALPHAR, ALPHAI, BETA, VL, &ldvl_, VR, &ldvr_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* ALPHA,
    std::complex<float>* BETA,
    std::complex<float>* VL, int64_t ldvl,
    std::complex<float>* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvr) > std::numeric_limits<blas_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldvl_ = (blas_int) ldvl;
    blas_int ldvr_ = (blas_int) ldvr;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_cggev( &jobvl_, &jobvr_, &n_, A, &lda_, B, &ldb_, ALPHA, BETA, VL, &ldvl_, VR, &ldvr_, qry_work, &ineg_one, qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );
    std::vector< float > rwork( (8*n) );

    LAPACK_cggev( &jobvl_, &jobvr_, &n_, A, &lda_, B, &ldb_, ALPHA, BETA, VL, &ldvl_, VR, &ldvr_, &work[0], &lwork_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* ALPHA,
    std::complex<double>* BETA,
    std::complex<double>* VL, int64_t ldvl,
    std::complex<double>* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvr) > std::numeric_limits<blas_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldvl_ = (blas_int) ldvl;
    blas_int ldvr_ = (blas_int) ldvr;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_zggev( &jobvl_, &jobvr_, &n_, A, &lda_, B, &ldb_, ALPHA, BETA, VL, &ldvl_, VR, &ldvr_, qry_work, &ineg_one, qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );
    std::vector< double > rwork( (8*n) );

    LAPACK_zggev( &jobvl_, &jobvr_, &n_, A, &lda_, B, &ldb_, ALPHA, BETA, VL, &ldvl_, VR, &ldvr_, &work[0], &lwork_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
