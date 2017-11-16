#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t gesvd(
    lapack::Job jobu, lapack::Job jobvt, int64_t m, int64_t n,
    float* A, int64_t lda,
    float* S,
    float* U, int64_t ldu,
    float* VT, int64_t ldvt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvt) > std::numeric_limits<blas_int>::max() );
    }
    char jobu_ = job2char( jobu );
    char jobvt_ = job2char( jobvt );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldvt_ = (blas_int) ldvt;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_sgesvd( &jobu_, &jobvt_, &m_, &n_, A, &lda_, S, U, &ldu_, VT, &ldvt_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_sgesvd( &jobu_, &jobvt_, &m_, &n_, A, &lda_, S, U, &ldu_, VT, &ldvt_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gesvd(
    lapack::Job jobu, lapack::Job jobvt, int64_t m, int64_t n,
    double* A, int64_t lda,
    double* S,
    double* U, int64_t ldu,
    double* VT, int64_t ldvt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvt) > std::numeric_limits<blas_int>::max() );
    }
    char jobu_ = job2char( jobu );
    char jobvt_ = job2char( jobvt );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldvt_ = (blas_int) ldvt;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dgesvd( &jobu_, &jobvt_, &m_, &n_, A, &lda_, S, U, &ldu_, VT, &ldvt_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dgesvd( &jobu_, &jobvt_, &m_, &n_, A, &lda_, S, U, &ldu_, VT, &ldvt_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gesvd(
    lapack::Job jobu, lapack::Job jobvt, int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda,
    float* S,
    std::complex<float>* U, int64_t ldu,
    std::complex<float>* VT, int64_t ldvt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvt) > std::numeric_limits<blas_int>::max() );
    }
    char jobu_ = job2char( jobu );
    char jobvt_ = job2char( jobvt );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldvt_ = (blas_int) ldvt;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_cgesvd( &jobu_, &jobvt_, &m_, &n_, A, &lda_, S, U, &ldu_, VT, &ldvt_, qry_work, &ineg_one, qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );
    std::vector< float > rwork( (5*min(m,n)) );

    LAPACK_cgesvd( &jobu_, &jobvt_, &m_, &n_, A, &lda_, S, U, &ldu_, VT, &ldvt_, &work[0], &lwork_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gesvd(
    lapack::Job jobu, lapack::Job jobvt, int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda,
    double* S,
    std::complex<double>* U, int64_t ldu,
    std::complex<double>* VT, int64_t ldvt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldvt) > std::numeric_limits<blas_int>::max() );
    }
    char jobu_ = job2char( jobu );
    char jobvt_ = job2char( jobvt );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldvt_ = (blas_int) ldvt;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_zgesvd( &jobu_, &jobvt_, &m_, &n_, A, &lda_, S, U, &ldu_, VT, &ldvt_, qry_work, &ineg_one, qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );
    std::vector< double > rwork( (5*min(m,n)) );

    LAPACK_zgesvd( &jobu_, &jobvt_, &m_, &n_, A, &lda_, S, U, &ldu_, VT, &ldvt_, &work[0], &lwork_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
