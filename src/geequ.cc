#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t geequ(
    int64_t m, int64_t n,
    float const* A, int64_t lda,
    float* R,
    float* C,
    float* rowcnd,
    float* colcnd,
    float* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    LAPACK_sgeequ( &m_, &n_, A, &lda_, R, C, rowcnd, colcnd, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geequ(
    int64_t m, int64_t n,
    double const* A, int64_t lda,
    double* R,
    double* C,
    double* rowcnd,
    double* colcnd,
    double* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    LAPACK_dgeequ( &m_, &n_, A, &lda_, R, C, rowcnd, colcnd, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geequ(
    int64_t m, int64_t n,
    std::complex<float> const* A, int64_t lda,
    float* R,
    float* C,
    float* rowcnd,
    float* colcnd,
    float* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    LAPACK_cgeequ( &m_, &n_, A, &lda_, R, C, rowcnd, colcnd, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t geequ(
    int64_t m, int64_t n,
    std::complex<double> const* A, int64_t lda,
    double* R,
    double* C,
    double* rowcnd,
    double* colcnd,
    double* amax )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    LAPACK_zgeequ( &m_, &n_, A, &lda_, R, C, rowcnd, colcnd, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
