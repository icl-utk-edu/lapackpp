#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t gbequb(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    float const* AB, int64_t ldab,
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
        throw_if_( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int ldab_ = (blas_int) ldab;
    blas_int info_ = 0;

    LAPACK_sgbequb( &m_, &n_, &kl_, &ku_, AB, &ldab_, R, C, rowcnd, colcnd, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gbequb(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    double const* AB, int64_t ldab,
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
        throw_if_( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int ldab_ = (blas_int) ldab;
    blas_int info_ = 0;

    LAPACK_dgbequb( &m_, &n_, &kl_, &ku_, AB, &ldab_, R, C, rowcnd, colcnd, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gbequb(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    std::complex<float> const* AB, int64_t ldab,
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
        throw_if_( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int ldab_ = (blas_int) ldab;
    blas_int info_ = 0;

    LAPACK_cgbequb( &m_, &n_, &kl_, &ku_, AB, &ldab_, R, C, rowcnd, colcnd, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gbequb(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    std::complex<double> const* AB, int64_t ldab,
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
        throw_if_( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int ldab_ = (blas_int) ldab;
    blas_int info_ = 0;

    LAPACK_zgbequb( &m_, &n_, &kl_, &ku_, AB, &ldab_, R, C, rowcnd, colcnd, amax, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
