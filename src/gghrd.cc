#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t gghrd(
    lapack::CompQ compq, lapack::CompQ compz, int64_t n, int64_t ilo, int64_t ihi,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    float* Q, int64_t ldq,
    float* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char compq_ = compq2char( compq );
    char compz_ = compq2char( compz );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    LAPACK_sgghrd( &compq_, &compz_, &n_, &ilo_, &ihi_, A, &lda_, B, &ldb_, Q, &ldq_, Z, &ldz_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gghrd(
    lapack::CompQ compq, lapack::CompQ compz, int64_t n, int64_t ilo, int64_t ihi,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    double* Q, int64_t ldq,
    double* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char compq_ = compq2char( compq );
    char compz_ = compq2char( compz );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    LAPACK_dgghrd( &compq_, &compz_, &n_, &ilo_, &ihi_, A, &lda_, B, &ldb_, Q, &ldq_, Z, &ldz_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gghrd(
    lapack::CompQ compq, lapack::CompQ compz, int64_t n, int64_t ilo, int64_t ihi,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* Q, int64_t ldq,
    std::complex<float>* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char compq_ = compq2char( compq );
    char compz_ = compq2char( compz );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    LAPACK_cgghrd( &compq_, &compz_, &n_, &ilo_, &ihi_, A, &lda_, B, &ldb_, Q, &ldq_, Z, &ldz_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gghrd(
    lapack::CompQ compq, lapack::CompQ compz, int64_t n, int64_t ilo, int64_t ihi,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* Q, int64_t ldq,
    std::complex<double>* Z, int64_t ldz )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldq) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    char compq_ = compq2char( compq );
    char compz_ = compq2char( compz );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldq_ = (blas_int) ldq;
    blas_int ldz_ = (blas_int) ldz;
    blas_int info_ = 0;

    LAPACK_zgghrd( &compq_, &compz_, &n_, &ilo_, &ihi_, A, &lda_, B, &ldb_, Q, &ldq_, Z, &ldz_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
