#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t gees(
    lapack::Job jobvs, lapack::Sort sort, LAPACK_S_SELECT2 select, int64_t n,
    float* A, int64_t lda,
    int64_t* sdim,
    float* WR,
    float* WI,
    float* VS, int64_t ldvs )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvs) > std::numeric_limits<blas_int>::max() );
    }
    char jobvs_ = job2char( jobvs );
    char sort_ = sort2char( sort );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int sdim_ = (blas_int) *sdim;
    blas_int ldvs_ = (blas_int) ldvs;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int qry_bwork[1];
    blas_int ineg_one = -1;
    LAPACK_sgees( &jobvs_, &sort_, select, &n_, A, &lda_, &sdim_, WR, WI, VS, &ldvs_, qry_work, &ineg_one, qry_bwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );
    std::vector< blas_int > bwork( (n) );

    LAPACK_sgees( &jobvs_, &sort_, select, &n_, A, &lda_, &sdim_, WR, WI, VS, &ldvs_, &work[0], &lwork_, &bwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gees(
    lapack::Job jobvs, lapack::Sort sort, LAPACK_D_SELECT2 select, int64_t n,
    double* A, int64_t lda,
    int64_t* sdim,
    double* WR,
    double* WI,
    double* VS, int64_t ldvs )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvs) > std::numeric_limits<blas_int>::max() );
    }
    char jobvs_ = job2char( jobvs );
    char sort_ = sort2char( sort );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int sdim_ = (blas_int) *sdim;
    blas_int ldvs_ = (blas_int) ldvs;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int qry_bwork[1];
    blas_int ineg_one = -1;
    LAPACK_dgees( &jobvs_, &sort_, select, &n_, A, &lda_, &sdim_, WR, WI, VS, &ldvs_, qry_work, &ineg_one, qry_bwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );
    std::vector< blas_int > bwork( (n) );

    LAPACK_dgees( &jobvs_, &sort_, select, &n_, A, &lda_, &sdim_, WR, WI, VS, &ldvs_, &work[0], &lwork_, &bwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gees(
    lapack::Job jobvs, lapack::Sort sort, LAPACK_C_SELECT1 select, int64_t n,
    std::complex<float>* A, int64_t lda,
    int64_t* sdim,
    std::complex<float>* W,
    std::complex<float>* VS, int64_t ldvs )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvs) > std::numeric_limits<blas_int>::max() );
    }
    char jobvs_ = job2char( jobvs );
    char sort_ = sort2char( sort );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int sdim_ = (blas_int) *sdim;
    blas_int ldvs_ = (blas_int) ldvs;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    blas_int qry_bwork[1];
    blas_int ineg_one = -1;
    LAPACK_cgees( &jobvs_, &sort_, select, &n_, A, &lda_, &sdim_, W, VS, &ldvs_, qry_work, &ineg_one, qry_rwork, qry_bwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );
    std::vector< float > rwork( (n) );
    std::vector< blas_int > bwork( (n) );

    LAPACK_cgees( &jobvs_, &sort_, select, &n_, A, &lda_, &sdim_, W, VS, &ldvs_, &work[0], &lwork_, &rwork[0], &bwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gees(
    lapack::Job jobvs, lapack::Sort sort, LAPACK_Z_SELECT1 select, int64_t n,
    std::complex<double>* A, int64_t lda,
    int64_t* sdim,
    std::complex<double>* W,
    std::complex<double>* VS, int64_t ldvs )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvs) > std::numeric_limits<blas_int>::max() );
    }
    char jobvs_ = job2char( jobvs );
    char sort_ = sort2char( sort );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int sdim_ = (blas_int) *sdim;
    blas_int ldvs_ = (blas_int) ldvs;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    blas_int qry_bwork[1];
    blas_int ineg_one = -1;
    LAPACK_zgees( &jobvs_, &sort_, select, &n_, A, &lda_, &sdim_, W, VS, &ldvs_, qry_work, &ineg_one, qry_rwork, qry_bwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );
    std::vector< double > rwork( (n) );
    std::vector< blas_int > bwork( (n) );

    LAPACK_zgees( &jobvs_, &sort_, select, &n_, A, &lda_, &sdim_, W, VS, &ldvs_, &work[0], &lwork_, &rwork[0], &bwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *sdim = sdim_;
    return info_;
}

}  // namespace lapack
