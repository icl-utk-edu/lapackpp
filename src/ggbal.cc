#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t ggbal(
    lapack::Job job, int64_t n,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    int64_t* ilo,
    int64_t* ihi,
    float* LSCALE,
    float* RSCALE )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ilo_ = (blas_int) *ilo;
    blas_int ihi_ = (blas_int) *ihi;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (lwork) );

    LAPACK_sggbal( &job_, &n_, A, &lda_, B, &ldb_, &ilo_, &ihi_, LSCALE, RSCALE, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbal(
    lapack::Job job, int64_t n,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    int64_t* ilo,
    int64_t* ihi,
    double* LSCALE,
    double* RSCALE )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ilo_ = (blas_int) *ilo;
    blas_int ihi_ = (blas_int) *ihi;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (lwork) );

    LAPACK_dggbal( &job_, &n_, A, &lda_, B, &ldb_, &ilo_, &ihi_, LSCALE, RSCALE, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbal(
    lapack::Job job, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    int64_t* ilo,
    int64_t* ihi,
    float* LSCALE,
    float* RSCALE )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ilo_ = (blas_int) *ilo;
    blas_int ihi_ = (blas_int) *ihi;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (lwork) );

    LAPACK_cggbal( &job_, &n_, A, &lda_, B, &ldb_, &ilo_, &ihi_, LSCALE, RSCALE, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbal(
    lapack::Job job, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    int64_t* ilo,
    int64_t* ihi,
    double* LSCALE,
    double* RSCALE )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ilo_ = (blas_int) *ilo;
    blas_int ihi_ = (blas_int) *ihi;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (lwork) );

    LAPACK_zggbal( &job_, &n_, A, &lda_, B, &ldb_, &ilo_, &ihi_, LSCALE, RSCALE, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

}  // namespace lapack
