#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t gebal(
    lapack::Balance balance, int64_t n,
    float* A, int64_t lda,
    int64_t* ilo,
    int64_t* ihi,
    float* SCALE )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ilo_ = (blas_int) *ilo;
    blas_int ihi_ = (blas_int) *ihi;
    blas_int info_ = 0;

    LAPACK_sgebal( &balance_, &n_, A, &lda_, &ilo_, &ihi_, SCALE, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gebal(
    lapack::Balance balance, int64_t n,
    double* A, int64_t lda,
    int64_t* ilo,
    int64_t* ihi,
    double* SCALE )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ilo_ = (blas_int) *ilo;
    blas_int ihi_ = (blas_int) *ihi;
    blas_int info_ = 0;

    LAPACK_dgebal( &balance_, &n_, A, &lda_, &ilo_, &ihi_, SCALE, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gebal(
    lapack::Balance balance, int64_t n,
    std::complex<float>* A, int64_t lda,
    int64_t* ilo,
    int64_t* ihi,
    float* SCALE )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ilo_ = (blas_int) *ilo;
    blas_int ihi_ = (blas_int) *ihi;
    blas_int info_ = 0;

    LAPACK_cgebal( &balance_, &n_, A, &lda_, &ilo_, &ihi_, SCALE, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gebal(
    lapack::Balance balance, int64_t n,
    std::complex<double>* A, int64_t lda,
    int64_t* ilo,
    int64_t* ihi,
    double* SCALE )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ilo_ = (blas_int) *ilo;
    blas_int ihi_ = (blas_int) *ihi;
    blas_int info_ = 0;

    LAPACK_zgebal( &balance_, &n_, A, &lda_, &ilo_, &ihi_, SCALE, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    *ilo = ilo_;
    *ihi = ihi_;
    return info_;
}

}  // namespace lapack
