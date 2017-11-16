#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t gbtrf(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    float* AB, int64_t ldab,
    int64_t* ipiv )
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
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( (min(m,n)) );
        blas_int* ipiv_ptr = &ipiv_[0];
    #else
        blas_int* ipiv_ptr = ipiv;
    #endif
    blas_int info_ = 0;

    LAPACK_sgbtrf( &m_, &n_, &kl_, &ku_, AB, &ldab_, ipiv_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gbtrf(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    double* AB, int64_t ldab,
    int64_t* ipiv )
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
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( (min(m,n)) );
        blas_int* ipiv_ptr = &ipiv_[0];
    #else
        blas_int* ipiv_ptr = ipiv;
    #endif
    blas_int info_ = 0;

    LAPACK_dgbtrf( &m_, &n_, &kl_, &ku_, AB, &ldab_, ipiv_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gbtrf(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    std::complex<float>* AB, int64_t ldab,
    int64_t* ipiv )
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
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( (min(m,n)) );
        blas_int* ipiv_ptr = &ipiv_[0];
    #else
        blas_int* ipiv_ptr = ipiv;
    #endif
    blas_int info_ = 0;

    LAPACK_cgbtrf( &m_, &n_, &kl_, &ku_, AB, &ldab_, ipiv_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gbtrf(
    int64_t m, int64_t n, int64_t kl, int64_t ku,
    std::complex<double>* AB, int64_t ldab,
    int64_t* ipiv )
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
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( (min(m,n)) );
        blas_int* ipiv_ptr = &ipiv_[0];
    #else
        blas_int* ipiv_ptr = ipiv;
    #endif
    blas_int info_ = 0;

    LAPACK_zgbtrf( &m_, &n_, &kl_, &ku_, AB, &ldab_, ipiv_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

}  // namespace lapack
