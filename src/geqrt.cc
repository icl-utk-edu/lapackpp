#include "lapack.hh"
#include "lapack_fortran.h"

#if LAPACK_VERSION_MAJOR >= 3 && LAPACK_VERSION_MINOR >= 4  // >= 3.4

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t geqrt(
    int64_t m, int64_t n, int64_t nb,
    float* A, int64_t lda,
    float* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(nb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int nb_ = (blas_int) nb;
    blas_int lda_ = (blas_int) lda;
    blas_int ldt_ = (blas_int) ldt;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (nb*n) );

    LAPACK_sgeqrt( &m_, &n_, &nb_, A, &lda_, T, &ldt_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geqrt(
    int64_t m, int64_t n, int64_t nb,
    double* A, int64_t lda,
    double* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(nb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int nb_ = (blas_int) nb;
    blas_int lda_ = (blas_int) lda;
    blas_int ldt_ = (blas_int) ldt;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (nb*n) );

    LAPACK_dgeqrt( &m_, &n_, &nb_, A, &lda_, T, &ldt_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geqrt(
    int64_t m, int64_t n, int64_t nb,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(nb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int nb_ = (blas_int) nb;
    blas_int lda_ = (blas_int) lda;
    blas_int ldt_ = (blas_int) ldt;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (nb*n) );

    LAPACK_cgeqrt( &m_, &n_, &nb_, A, &lda_, T, &ldt_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geqrt(
    int64_t m, int64_t n, int64_t nb,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(nb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int nb_ = (blas_int) nb;
    blas_int lda_ = (blas_int) lda;
    blas_int ldt_ = (blas_int) ldt;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (nb*n) );

    LAPACK_zgeqrt( &m_, &n_, &nb_, A, &lda_, T, &ldt_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.4
