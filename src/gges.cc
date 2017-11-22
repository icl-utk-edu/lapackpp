#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t gges(
    lapack::Job jobvsl, lapack::Job jobvsr, lapack::Sort sort, lapack_s_select3 select, int64_t n,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    int64_t* sdim,
    float* ALPHAR,
    float* ALPHAI,
    float* BETA,
    float* VSL, int64_t ldvsl,
    float* VSR, int64_t ldvsr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvsl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvsr) > std::numeric_limits<blas_int>::max() );
    }
    char jobvsl_ = job2char( jobvsl );
    char jobvsr_ = job2char( jobvsr );
    char sort_ = sort2char( sort );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int sdim_ = (blas_int) *sdim;
    blas_int ldvsl_ = (blas_int) ldvsl;
    blas_int ldvsr_ = (blas_int) ldvsr;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int qry_bwork[1];
    blas_int ineg_one = -1;
    LAPACK_sgges( &jobvsl_, &jobvsr_, &sort_, select, &n_, A, &lda_, B, &ldb_, &sdim_, ALPHAR, ALPHAI, BETA, VSL, &ldvsl_, VSR, &ldvsr_, qry_work, &ineg_one, qry_bwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );
    std::vector< blas_int > bwork( (n) );

    LAPACK_sgges( &jobvsl_, &jobvsr_, &sort_, select, &n_, A, &lda_, B, &ldb_, &sdim_, ALPHAR, ALPHAI, BETA, VSL, &ldvsl_, VSR, &ldvsr_, &work[0], &lwork_, &bwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gges(
    lapack::Job jobvsl, lapack::Job jobvsr, lapack::Sort sort, lapack_d_select3 select, int64_t n,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    int64_t* sdim,
    double* ALPHAR,
    double* ALPHAI,
    double* BETA,
    double* VSL, int64_t ldvsl,
    double* VSR, int64_t ldvsr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvsl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvsr) > std::numeric_limits<blas_int>::max() );
    }
    char jobvsl_ = job2char( jobvsl );
    char jobvsr_ = job2char( jobvsr );
    char sort_ = sort2char( sort );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int sdim_ = (blas_int) *sdim;
    blas_int ldvsl_ = (blas_int) ldvsl;
    blas_int ldvsr_ = (blas_int) ldvsr;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int qry_bwork[1];
    blas_int ineg_one = -1;
    LAPACK_dgges( &jobvsl_, &jobvsr_, &sort_, select, &n_, A, &lda_, B, &ldb_, &sdim_, ALPHAR, ALPHAI, BETA, VSL, &ldvsl_, VSR, &ldvsr_, qry_work, &ineg_one, qry_bwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );
    std::vector< blas_int > bwork( (n) );

    LAPACK_dgges( &jobvsl_, &jobvsr_, &sort_, select, &n_, A, &lda_, B, &ldb_, &sdim_, ALPHAR, ALPHAI, BETA, VSL, &ldvsl_, VSR, &ldvsr_, &work[0], &lwork_, &bwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gges(
    lapack::Job jobvsl, lapack::Job jobvsr, lapack::Sort sort, lapack_c_select2 select, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    int64_t* sdim,
    std::complex<float>* ALPHA,
    std::complex<float>* BETA,
    std::complex<float>* VSL, int64_t ldvsl,
    std::complex<float>* VSR, int64_t ldvsr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvsl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvsr) > std::numeric_limits<blas_int>::max() );
    }
    char jobvsl_ = job2char( jobvsl );
    char jobvsr_ = job2char( jobvsr );
    char sort_ = sort2char( sort );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int sdim_ = (blas_int) *sdim;
    blas_int ldvsl_ = (blas_int) ldvsl;
    blas_int ldvsr_ = (blas_int) ldvsr;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    blas_int qry_bwork[1];
    blas_int ineg_one = -1;
    LAPACK_cgges( &jobvsl_, &jobvsr_, &sort_, select, &n_, A, &lda_, B, &ldb_, &sdim_, ALPHA, BETA, VSL, &ldvsl_, VSR, &ldvsr_, qry_work, &ineg_one, qry_rwork, qry_bwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );
    std::vector< float > rwork( (8*n) );
    std::vector< blas_int > bwork( (n) );

    LAPACK_cgges( &jobvsl_, &jobvsr_, &sort_, select, &n_, A, &lda_, B, &ldb_, &sdim_, ALPHA, BETA, VSL, &ldvsl_, VSR, &ldvsr_, &work[0], &lwork_, &rwork[0], &bwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gges(
    lapack::Job jobvsl, lapack::Job jobvsr, lapack::Sort sort, lapack_z_select2 select, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    int64_t* sdim,
    std::complex<double>* ALPHA,
    std::complex<double>* BETA,
    std::complex<double>* VSL, int64_t ldvsl,
    std::complex<double>* VSR, int64_t ldvsr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvsl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvsr) > std::numeric_limits<blas_int>::max() );
    }
    char jobvsl_ = job2char( jobvsl );
    char jobvsr_ = job2char( jobvsr );
    char sort_ = sort2char( sort );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int sdim_ = (blas_int) *sdim;
    blas_int ldvsl_ = (blas_int) ldvsl;
    blas_int ldvsr_ = (blas_int) ldvsr;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    blas_int qry_bwork[1];
    blas_int ineg_one = -1;
    LAPACK_zgges( &jobvsl_, &jobvsr_, &sort_, select, &n_, A, &lda_, B, &ldb_, &sdim_, ALPHA, BETA, VSL, &ldvsl_, VSR, &ldvsr_, qry_work, &ineg_one, qry_rwork, qry_bwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );
    std::vector< double > rwork( (8*n) );
    std::vector< blas_int > bwork( (n) );

    LAPACK_zgges( &jobvsl_, &jobvsr_, &sort_, select, &n_, A, &lda_, B, &ldb_, &sdim_, ALPHA, BETA, VSL, &ldvsl_, VSR, &ldvsr_, &work[0], &lwork_, &rwork[0], &bwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

}  // namespace lapack
