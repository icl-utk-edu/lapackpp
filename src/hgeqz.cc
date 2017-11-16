#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t hgeqz(
    lapack::Job job, lapack::CompQ compq, lapack::CompQ compz, int64_t n, int64_t ilo, int64_t ihi,
    float* H, int64_t ldh,
    float* T, int64_t ldt,
    float* ALPHAR,
    float* ALPHAI,
    float* BETA,
    float* Q, int64_t ldq,
    float* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldh) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    char compq_ = compq2char( compq );
    char compz_ = compq2char( compz );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int ldh_ = (blas_int) ldh;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_shgeqz( &job_, &compq_, &compz_, &n_, &ilo_, &ihi_, H, &ldh_, T, &ldt_, ALPHAR, ALPHAI, BETA, Q, &ldq_, Z, &ldz_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_shgeqz( &job_, &compq_, &compz_, &n_, &ilo_, &ihi_, H, &ldh_, T, &ldt_, ALPHAR, ALPHAI, BETA, Q, &ldq_, Z, &ldz_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t hgeqz(
    lapack::Job job, lapack::CompQ compq, lapack::CompQ compz, int64_t n, int64_t ilo, int64_t ihi,
    double* H, int64_t ldh,
    double* T, int64_t ldt,
    double* ALPHAR,
    double* ALPHAI,
    double* BETA,
    double* Q, int64_t ldq,
    double* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldh) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    char compq_ = compq2char( compq );
    char compz_ = compq2char( compz );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int ldh_ = (blas_int) ldh;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dhgeqz( &job_, &compq_, &compz_, &n_, &ilo_, &ihi_, H, &ldh_, T, &ldt_, ALPHAR, ALPHAI, BETA, Q, &ldq_, Z, &ldz_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dhgeqz( &job_, &compq_, &compz_, &n_, &ilo_, &ihi_, H, &ldh_, T, &ldt_, ALPHAR, ALPHAI, BETA, Q, &ldq_, Z, &ldz_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t hgeqz(
    lapack::Job job, lapack::CompQ compq, lapack::CompQ compz, int64_t n, int64_t ilo, int64_t ihi,
    std::complex<float>* H, int64_t ldh,
    std::complex<float>* T, int64_t ldt,
    std::complex<float>* ALPHA,
    std::complex<float>* BETA,
    std::complex<float>* Q, int64_t ldq,
    std::complex<float>* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldh) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    char compq_ = compq2char( compq );
    char compz_ = compq2char( compz );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int ldh_ = (blas_int) ldh;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    float qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_chgeqz( &job_, &compq_, &compz_, &n_, &ilo_, &ihi_, H, &ldh_, T, &ldt_, ALPHA, BETA, Q, &ldq_, Z, &ldz_, qry_work, &ineg_one, qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );
    std::vector< float > rwork( (n) );

    LAPACK_chgeqz( &job_, &compq_, &compz_, &n_, &ilo_, &ihi_, H, &ldh_, T, &ldt_, ALPHA, BETA, Q, &ldq_, Z, &ldz_, &work[0], &lwork_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t hgeqz(
    lapack::Job job, lapack::CompQ compq, lapack::CompQ compz, int64_t n, int64_t ilo, int64_t ihi,
    std::complex<double>* H, int64_t ldh,
    std::complex<double>* T, int64_t ldt,
    std::complex<double>* ALPHA,
    std::complex<double>* BETA,
    std::complex<double>* Q, int64_t ldq,
    std::complex<double>* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldh) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    char compq_ = compq2char( compq );
    char compz_ = compq2char( compz );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int ldh_ = (blas_int) ldh;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    double qry_rwork[1];
    blas_int ineg_one = -1;
    LAPACK_zhgeqz( &job_, &compq_, &compz_, &n_, &ilo_, &ihi_, H, &ldh_, T, &ldt_, ALPHA, BETA, Q, &ldq_, Z, &ldz_, qry_work, &ineg_one, qry_rwork, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );
    std::vector< double > rwork( (n) );

    LAPACK_zhgeqz( &job_, &compq_, &compz_, &n_, &ilo_, &ihi_, H, &ldh_, T, &ldt_, ALPHA, BETA, Q, &ldq_, Z, &ldz_, &work[0], &lwork_, &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
