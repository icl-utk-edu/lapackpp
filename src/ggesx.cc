#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t ggesx(
    lapack::Job jobvsl, lapack::Job jobvsr, lapack::Sort sort, LAPACK_S_SELECT3 select, lapack::Sense sense, int64_t n,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    int64_t* sdim,
    float* ALPHAR,
    float* ALPHAI,
    float* BETA,
    float* VSL, int64_t ldvsl,
    float* VSR, int64_t ldvsr,
    float* RCONDE,
    float* RCONDV )
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
    char sense_ = sense2char( sense );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int sdim_ = (blas_int) *sdim;
    blas_int ldvsl_ = (blas_int) ldvsl;
    blas_int ldvsr_ = (blas_int) ldvsr;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int qry_iwork[1];
    blas_int qry_bwork[1];
    blas_int ineg_one = -1;
    LAPACK_sggesx( &jobvsl_, &jobvsr_, &sort_, select, &sense_, &n_, A, &lda_, B, &ldb_, &sdim_, ALPHAR, ALPHAI, BETA, VSL, &ldvsl_, VSR, &ldvsr_, RCONDE, RCONDV, qry_work, &ineg_one, qry_iwork, &ineg_one, qry_bwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);
    blas_int liwork_ = real(qry_iwork[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );
    std::vector< blas_int > iwork( liwork_ );
    std::vector< blas_int > bwork( (n) );

    LAPACK_sggesx( &jobvsl_, &jobvsr_, &sort_, select, &sense_, &n_, A, &lda_, B, &ldb_, &sdim_, ALPHAR, ALPHAI, BETA, VSL, &ldvsl_, VSR, &ldvsr_, RCONDE, RCONDV, &work[0], &lwork_, &iwork[0], &liwork_, &bwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggesx(
    lapack::Job jobvsl, lapack::Job jobvsr, lapack::Sort sort, LAPACK_D_SELECT3 select, lapack::Sense sense, int64_t n,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    int64_t* sdim,
    double* ALPHAR,
    double* ALPHAI,
    double* BETA,
    double* VSL, int64_t ldvsl,
    double* VSR, int64_t ldvsr,
    double* RCONDE,
    double* RCONDV )
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
    char sense_ = sense2char( sense );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int sdim_ = (blas_int) *sdim;
    blas_int ldvsl_ = (blas_int) ldvsl;
    blas_int ldvsr_ = (blas_int) ldvsr;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int qry_iwork[1];
    blas_int qry_bwork[1];
    blas_int ineg_one = -1;
    LAPACK_dggesx( &jobvsl_, &jobvsr_, &sort_, select, &sense_, &n_, A, &lda_, B, &ldb_, &sdim_, ALPHAR, ALPHAI, BETA, VSL, &ldvsl_, VSR, &ldvsr_, RCONDE, RCONDV, qry_work, &ineg_one, qry_iwork, &ineg_one, qry_bwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);
    blas_int liwork_ = real(qry_iwork[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );
    std::vector< blas_int > iwork( liwork_ );
    std::vector< blas_int > bwork( (n) );

    LAPACK_dggesx( &jobvsl_, &jobvsr_, &sort_, select, &sense_, &n_, A, &lda_, B, &ldb_, &sdim_, ALPHAR, ALPHAI, BETA, VSL, &ldvsl_, VSR, &ldvsr_, RCONDE, RCONDV, &work[0], &lwork_, &iwork[0], &liwork_, &bwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggesx(
    lapack::Job jobvsl, lapack::Job jobvsr, lapack::Sort sort, LAPACK_C_SELECT2 select, lapack::Sense sense, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    int64_t* sdim,
    std::complex<float>* ALPHA,
    std::complex<float>* BETA,
    std::complex<float>* VSL, int64_t ldvsl,
    std::complex<float>* VSR, int64_t ldvsr,
    float* RCONDE,
    float* RCONDV )
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
    char sense_ = sense2char( sense );
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
    blas_int qry_iwork[1];
    blas_int qry_bwork[1];
    blas_int ineg_one = -1;
    LAPACK_cggesx( &jobvsl_, &jobvsr_, &sort_, select, &sense_, &n_, A, &lda_, B, &ldb_, &sdim_, ALPHA, BETA, VSL, &ldvsl_, VSR, &ldvsr_, RCONDE, RCONDV, qry_work, &ineg_one, qry_rwork, qry_iwork, &ineg_one, qry_bwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);
    blas_int liwork_ = real(qry_iwork[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );
    std::vector< float > rwork( (8*n) );
    std::vector< blas_int > iwork( liwork_ );
    std::vector< blas_int > bwork( (n) );

    LAPACK_cggesx( &jobvsl_, &jobvsr_, &sort_, select, &sense_, &n_, A, &lda_, B, &ldb_, &sdim_, ALPHA, BETA, VSL, &ldvsl_, VSR, &ldvsr_, RCONDE, RCONDV, &work[0], &lwork_, &rwork[0], &iwork[0], &liwork_, &bwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggesx(
    lapack::Job jobvsl, lapack::Job jobvsr, lapack::Sort sort, LAPACK_Z_SELECT2 select, lapack::Sense sense, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    int64_t* sdim,
    std::complex<double>* ALPHA,
    std::complex<double>* BETA,
    std::complex<double>* VSL, int64_t ldvsl,
    std::complex<double>* VSR, int64_t ldvsr,
    double* RCONDE,
    double* RCONDV )
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
    char sense_ = sense2char( sense );
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
    blas_int qry_iwork[1];
    blas_int qry_bwork[1];
    blas_int ineg_one = -1;
    LAPACK_zggesx( &jobvsl_, &jobvsr_, &sort_, select, &sense_, &n_, A, &lda_, B, &ldb_, &sdim_, ALPHA, BETA, VSL, &ldvsl_, VSR, &ldvsr_, RCONDE, RCONDV, qry_work, &ineg_one, qry_rwork, qry_iwork, &ineg_one, qry_bwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);
    blas_int liwork_ = real(qry_iwork[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );
    std::vector< double > rwork( (8*n) );
    std::vector< blas_int > iwork( liwork_ );
    std::vector< blas_int > bwork( (n) );

    LAPACK_zggesx( &jobvsl_, &jobvsr_, &sort_, select, &sense_, &n_, A, &lda_, B, &ldb_, &sdim_, ALPHA, BETA, VSL, &ldvsl_, VSR, &ldvsr_, RCONDE, RCONDV, &work[0], &lwork_, &rwork[0], &iwork[0], &liwork_, &bwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

}  // namespace lapack
