#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t ggbal(
    lapack::Balance balance, int64_t n,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    int64_t* ilo,
    int64_t* ihi,
    float* lscale,
    float* rscale )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ilo_ = (blas_int) *ilo;
    blas_int ihi_ = (blas_int) *ihi;
    blas_int info_ = 0;

    // from docs
    int64_t lwork = (balance == Balance::Scale || balance == Balance::Both ? max( 1, 6*n ) : 1);

    // allocate workspace
    std::vector< float > work( (lwork) );

    LAPACK_sggbal(
        &balance_, &n_,
        A, &lda_,
        B, &ldb_, &ilo_, &ihi_,
        lscale,
        rscale,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbal(
    lapack::Balance balance, int64_t n,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    int64_t* ilo,
    int64_t* ihi,
    double* lscale,
    double* rscale )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ilo_ = (blas_int) *ilo;
    blas_int ihi_ = (blas_int) *ihi;
    blas_int info_ = 0;

    // from docs
    int64_t lwork = (balance == Balance::Scale || balance == Balance::Both ? max( 1, 6*n ) : 1);

    // allocate workspace
    std::vector< double > work( (lwork) );

    LAPACK_dggbal(
        &balance_, &n_,
        A, &lda_,
        B, &ldb_, &ilo_, &ihi_,
        lscale,
        rscale,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbal(
    lapack::Balance balance, int64_t n,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    int64_t* ilo,
    int64_t* ihi,
    float* lscale,
    float* rscale )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ilo_ = (blas_int) *ilo;
    blas_int ihi_ = (blas_int) *ihi;
    blas_int info_ = 0;

    // from docs
    int64_t lwork = (balance == Balance::Scale || balance == Balance::Both ? max( 1, 6*n ) : 1);

    // allocate workspace
    std::vector< float > work( (lwork) );

    LAPACK_cggbal(
        &balance_, &n_,
        (lapack_complex_float*) A, &lda_,
        (lapack_complex_float*) B, &ldb_, &ilo_, &ihi_,
        lscale,
        rscale,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbal(
    lapack::Balance balance, int64_t n,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    int64_t* ilo,
    int64_t* ihi,
    double* lscale,
    double* rscale )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ilo_ = (blas_int) *ilo;
    blas_int ihi_ = (blas_int) *ihi;
    blas_int info_ = 0;

    // from docs
    int64_t lwork = (balance == Balance::Scale || balance == Balance::Both ? max( 1, 6*n ) : 1);

    // allocate workspace
    std::vector< double > work( (lwork) );

    LAPACK_zggbal(
        &balance_, &n_,
        (lapack_complex_double*) A, &lda_,
        (lapack_complex_double*) B, &ldb_, &ilo_, &ihi_,
        lscale,
        rscale,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

}  // namespace lapack
