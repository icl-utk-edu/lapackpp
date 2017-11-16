#include "lapack.hh"
#include "lapack_fortran.h"

#if LAPACK_VERSION_MAJOR >= 3 && LAPACK_VERSION_MINOR >= 6  // >= v3.6

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t ggsvp3(
    lapack::Job jobu, lapack::Job jobv, lapack::JobQ jobq, int64_t m, int64_t p, int64_t n,
    float* A, int64_t lda,
    float* B, int64_t ldb, float tola, float tolb,
    int64_t* k,
    int64_t* l,
    float* U, int64_t ldu,
    float* V, int64_t ldv,
    float* Q, int64_t ldq,
    float* TAU )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(p) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char jobu_ = job2char( jobu );
    char jobv_ = job2char( jobv );
    char jobq_ = jobq2char( jobq );
    blas_int m_ = (blas_int) m;
    blas_int p_ = (blas_int) p;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int k_ = (blas_int) *k;
    blas_int l_ = (blas_int) *l;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldq_ = (blas_int) ldq;
    blas_int info_ = 0;

    // query for workspace size
    blas_int qry_iwork[1];
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_sggsvp3( &jobu_, &jobv_, &jobq_, &m_, &p_, &n_, A, &lda_, B, &ldb_, &tola, &tolb, &k_, &l_, U, &ldu_, V, &ldv_, Q, &ldq_, qry_iwork, TAU, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< blas_int > iwork( (n) );
    std::vector< float > work( lwork_ );

    LAPACK_sggsvp3( &jobu_, &jobv_, &jobq_, &m_, &p_, &n_, A, &lda_, B, &ldb_, &tola, &tolb, &k_, &l_, U, &ldu_, V, &ldv_, Q, &ldq_, &iwork[0], TAU, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *k = k_;
    *l = l_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggsvp3(
    lapack::Job jobu, lapack::Job jobv, lapack::JobQ jobq, int64_t m, int64_t p, int64_t n,
    double* A, int64_t lda,
    double* B, int64_t ldb, double tola, double tolb,
    int64_t* k,
    int64_t* l,
    double* U, int64_t ldu,
    double* V, int64_t ldv,
    double* Q, int64_t ldq,
    double* TAU )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(p) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char jobu_ = job2char( jobu );
    char jobv_ = job2char( jobv );
    char jobq_ = jobq2char( jobq );
    blas_int m_ = (blas_int) m;
    blas_int p_ = (blas_int) p;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int k_ = (blas_int) *k;
    blas_int l_ = (blas_int) *l;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldq_ = (blas_int) ldq;
    blas_int info_ = 0;

    // query for workspace size
    blas_int qry_iwork[1];
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dggsvp3( &jobu_, &jobv_, &jobq_, &m_, &p_, &n_, A, &lda_, B, &ldb_, &tola, &tolb, &k_, &l_, U, &ldu_, V, &ldv_, Q, &ldq_, qry_iwork, TAU, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< blas_int > iwork( (n) );
    std::vector< double > work( lwork_ );

    LAPACK_dggsvp3( &jobu_, &jobv_, &jobq_, &m_, &p_, &n_, A, &lda_, B, &ldb_, &tola, &tolb, &k_, &l_, U, &ldu_, V, &ldv_, Q, &ldq_, &iwork[0], TAU, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *k = k_;
    *l = l_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggsvp3(
    lapack::Job jobu, lapack::Job jobv, lapack::JobQ jobq, int64_t m, int64_t p, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb, float tola, float tolb,
    int64_t* k,
    int64_t* l,
    std::complex<float>* U, int64_t ldu,
    std::complex<float>* V, int64_t ldv,
    std::complex<float>* Q, int64_t ldq,
    std::complex<float>* TAU )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(p) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char jobu_ = job2char( jobu );
    char jobv_ = job2char( jobv );
    char jobq_ = jobq2char( jobq );
    blas_int m_ = (blas_int) m;
    blas_int p_ = (blas_int) p;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int k_ = (blas_int) *k;
    blas_int l_ = (blas_int) *l;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldq_ = (blas_int) ldq;
    blas_int info_ = 0;

    // query for workspace size
    blas_int qry_iwork[1];
    float qry_rwork[1];
    std::complex<float> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_cggsvp3( &jobu_, &jobv_, &jobq_, &m_, &p_, &n_, A, &lda_, B, &ldb_, &tola, &tolb, &k_, &l_, U, &ldu_, V, &ldv_, Q, &ldq_, qry_iwork, qry_rwork, TAU, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< blas_int > iwork( (n) );
    std::vector< float > rwork( (2*n) );
    std::vector< std::complex<float> > work( lwork_ );

    LAPACK_cggsvp3( &jobu_, &jobv_, &jobq_, &m_, &p_, &n_, A, &lda_, B, &ldb_, &tola, &tolb, &k_, &l_, U, &ldu_, V, &ldv_, Q, &ldq_, &iwork[0], &rwork[0], TAU, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *k = k_;
    *l = l_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggsvp3(
    lapack::Job jobu, lapack::Job jobv, lapack::JobQ jobq, int64_t m, int64_t p, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb, double tola, double tolb,
    int64_t* k,
    int64_t* l,
    std::complex<double>* U, int64_t ldu,
    std::complex<double>* V, int64_t ldv,
    std::complex<double>* Q, int64_t ldq,
    std::complex<double>* TAU )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(p) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldu) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
    }
    char jobu_ = job2char( jobu );
    char jobv_ = job2char( jobv );
    char jobq_ = jobq2char( jobq );
    blas_int m_ = (blas_int) m;
    blas_int p_ = (blas_int) p;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int k_ = (blas_int) *k;
    blas_int l_ = (blas_int) *l;
    blas_int ldu_ = (blas_int) ldu;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldq_ = (blas_int) ldq;
    blas_int info_ = 0;

    // query for workspace size
    blas_int qry_iwork[1];
    double qry_rwork[1];
    std::complex<double> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_zggsvp3( &jobu_, &jobv_, &jobq_, &m_, &p_, &n_, A, &lda_, B, &ldb_, &tola, &tolb, &k_, &l_, U, &ldu_, V, &ldv_, Q, &ldq_, qry_iwork, qry_rwork, TAU, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< blas_int > iwork( (n) );
    std::vector< double > rwork( (2*n) );
    std::vector< std::complex<double> > work( lwork_ );

    LAPACK_zggsvp3( &jobu_, &jobv_, &jobq_, &m_, &p_, &n_, A, &lda_, B, &ldb_, &tola, &tolb, &k_, &l_, U, &ldu_, V, &ldv_, Q, &ldq_, &iwork[0], &rwork[0], TAU, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *k = k_;
    *l = l_;
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= v3.6
