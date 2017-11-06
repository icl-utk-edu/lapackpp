#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t unghr(
    int64_t n, int64_t ilo, int64_t ihi,
    std::complex<float>* A, int64_t lda,
    std::complex<float> const* TAU )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_cunghr( &n_, &ilo_, &ihi_, A, &lda_, TAU, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );

    LAPACK_cunghr( &n_, &ilo_, &ihi_, A, &lda_, TAU, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t unghr(
    int64_t n, int64_t ilo, int64_t ihi,
    std::complex<double>* A, int64_t lda,
    std::complex<double> const* TAU )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_zunghr( &n_, &ilo_, &ihi_, A, &lda_, TAU, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );

    LAPACK_zunghr( &n_, &ilo_, &ihi_, A, &lda_, TAU, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
