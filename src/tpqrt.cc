#include "lapack.hh"
#include "lapack_fortran.h"

#if LAPACK_VERSION_MAJOR >= 3 && LAPACK_VERSION_MINOR >= 4  // >= 3.4

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t tpqrt(
    int64_t m, int64_t n, int64_t l, int64_t nb,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    float* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(l) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int l_ = (blas_int) l;
    blas_int nb_ = (blas_int) nb;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldt_ = (blas_int) ldt;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (nb*n) );

    LAPACK_stpqrt( &m_, &n_, &l_, &nb_, A, &lda_, B, &ldb_, T, &ldt_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tpqrt(
    int64_t m, int64_t n, int64_t l, int64_t nb,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    double* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(l) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int l_ = (blas_int) l;
    blas_int nb_ = (blas_int) nb;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldt_ = (blas_int) ldt;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (nb*n) );

    LAPACK_dtpqrt( &m_, &n_, &l_, &nb_, A, &lda_, B, &ldb_, T, &ldt_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tpqrt(
    int64_t m, int64_t n, int64_t l, int64_t nb,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(l) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int l_ = (blas_int) l;
    blas_int nb_ = (blas_int) nb;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldt_ = (blas_int) ldt;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (nb*n) );

    LAPACK_ctpqrt( &m_, &n_, &l_, &nb_, A, &lda_, B, &ldb_, T, &ldt_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t tpqrt(
    int64_t m, int64_t n, int64_t l, int64_t nb,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(l) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int l_ = (blas_int) l;
    blas_int nb_ = (blas_int) nb;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldt_ = (blas_int) ldt;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (nb*n) );

    LAPACK_ztpqrt( &m_, &n_, &l_, &nb_, A, &lda_, B, &ldb_, T, &ldt_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.4
