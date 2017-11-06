#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t hseqr(
    lapack::Job job, lapack::CompQ compz, int64_t n, int64_t ilo, int64_t ihi,
    float* H, int64_t ldh,
    float* WR,
    float* WI,
    float* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldh) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    char compz_ = compq2char( compz );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int ldh_ = (blas_int) ldh;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_shseqr( &job_, &compz_, &n_, &ilo_, &ihi_, H, &ldh_, WR, WI, Z, &ldz_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_shseqr( &job_, &compz_, &n_, &ilo_, &ihi_, H, &ldh_, WR, WI, Z, &ldz_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t hseqr(
    lapack::Job job, lapack::CompQ compz, int64_t n, int64_t ilo, int64_t ihi,
    double* H, int64_t ldh,
    double* WR,
    double* WI,
    double* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldh) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    char compz_ = compq2char( compz );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int ldh_ = (blas_int) ldh;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dhseqr( &job_, &compz_, &n_, &ilo_, &ihi_, H, &ldh_, WR, WI, Z, &ldz_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dhseqr( &job_, &compz_, &n_, &ilo_, &ihi_, H, &ldh_, WR, WI, Z, &ldz_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t hseqr(
    lapack::Job job, lapack::CompQ compz, int64_t n, int64_t ilo, int64_t ihi,
    std::complex<float>* H, int64_t ldh,
    std::complex<float>* W,
    std::complex<float>* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldh) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    char compz_ = compq2char( compz );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int ldh_ = (blas_int) ldh;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_chseqr( &job_, &compz_, &n_, &ilo_, &ihi_, H, &ldh_, W, Z, &ldz_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );

    LAPACK_chseqr( &job_, &compz_, &n_, &ilo_, &ihi_, H, &ldh_, W, Z, &ldz_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t hseqr(
    lapack::Job job, lapack::CompQ compz, int64_t n, int64_t ilo, int64_t ihi,
    std::complex<double>* H, int64_t ldh,
    std::complex<double>* W,
    std::complex<double>* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldh) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    char compz_ = compq2char( compz );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int ldh_ = (blas_int) ldh;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_zhseqr( &job_, &compz_, &n_, &ilo_, &ihi_, H, &ldh_, W, Z, &ldz_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );

    LAPACK_zhseqr( &job_, &compz_, &n_, &ilo_, &ihi_, H, &ldh_, W, Z, &ldz_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
