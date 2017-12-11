#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t geev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    float* A, int64_t lda,
    std::complex<float>* W,
    float* VL, int64_t ldvl,
    float* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvr) > std::numeric_limits<blas_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldvl_ = (blas_int) ldvl;
    blas_int ldvr_ = (blas_int) ldvr;
    blas_int info_ = 0;

    // split-complex representation
    std::vector< float > WR( max( 1, n ) );
    std::vector< float > WI( max( 1, n ) );

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_sgeev( &jobvl_, &jobvr_, &n_, A, &lda_, &WR[0], &WI[0], VL, &ldvl_, VR, &ldvr_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_sgeev( &jobvl_, &jobvr_, &n_, A, &lda_, &WR[0], &WI[0], VL, &ldvl_, VR, &ldvr_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    // merge split-complex representation
    for (int64_t i = 0; i < n; ++i) {
        W[i] = std::complex<float>( WR[i], WI[i] );
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    double* A, int64_t lda,
    std::complex<double>* W,
    double* VL, int64_t ldvl,
    double* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvr) > std::numeric_limits<blas_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldvl_ = (blas_int) ldvl;
    blas_int ldvr_ = (blas_int) ldvr;
    blas_int info_ = 0;

    // split-complex representation
    std::vector< double > WR( max( 1, n ) );
    std::vector< double > WI( max( 1, n ) );

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dgeev( &jobvl_, &jobvr_, &n_, A, &lda_, &WR[0], &WI[0], VL, &ldvl_, VR, &ldvr_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dgeev( &jobvl_, &jobvr_, &n_, A, &lda_, &WR[0], &WI[0], VL, &ldvl_, VR, &ldvr_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    // merge split-complex representation
    for (int64_t i = 0; i < n; ++i) {
        W[i] = std::complex<double>( WR[i], WI[i] );
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* W,
    std::complex<float>* VL, int64_t ldvl,
    std::complex<float>* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvr) > std::numeric_limits<blas_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldvl_ = (blas_int) ldvl;
    blas_int ldvr_ = (blas_int) ldvr;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_cgeev( &jobvl_, &jobvr_, &n_, A, &lda_, W, VL, &ldvl_, VR, &ldvr_, qry_work, &ineg_one, qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );
    std::vector< float > rwork( (2*n) );

    LAPACK_cgeev( &jobvl_, &jobvr_, &n_, A, &lda_, W, VL, &ldvl_, VR, &ldvr_, &work[0], &lwork_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geev(
    lapack::Job jobvl, lapack::Job jobvr, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* W,
    std::complex<double>* VL, int64_t ldvl,
    std::complex<double>* VR, int64_t ldvr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvr) > std::numeric_limits<blas_int>::max() );
    }
    char jobvl_ = job2char( jobvl );
    char jobvr_ = job2char( jobvr );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldvl_ = (blas_int) ldvl;
    blas_int ldvr_ = (blas_int) ldvr;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_zgeev( &jobvl_, &jobvr_, &n_, A, &lda_, W, VL, &ldvl_, VR, &ldvr_, qry_work, &ineg_one, qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );
    std::vector< double > rwork( (2*n) );

    LAPACK_zgeev( &jobvl_, &jobvr_, &n_, A, &lda_, W, VL, &ldvl_, VR, &ldvr_, &work[0], &lwork_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
