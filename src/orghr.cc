#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t orghr(
    int64_t n, int64_t ilo, int64_t ihi,
    float* A, int64_t lda,
    float const* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_sorghr(
        &n_, &ilo_, &ihi_,
        A, &lda_,
        tau,
        qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_sorghr(
        &n_, &ilo_, &ihi_,
        A, &lda_,
        tau,
        &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t orghr(
    int64_t n, int64_t ilo, int64_t ihi,
    double* A, int64_t lda,
    double const* tau )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dorghr(
        &n_, &ilo_, &ihi_,
        A, &lda_,
        tau,
        qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dorghr(
        &n_, &ilo_, &ihi_,
        A, &lda_,
        tau,
        &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
