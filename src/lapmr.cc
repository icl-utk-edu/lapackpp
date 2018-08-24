#include "lapack.hh"
#include "lapack_fortran.h"

#if LAPACK_VERSION >= 30300  // >= v3.3

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
void lapmr(
    bool forwrd, int64_t m, int64_t n,
    float* X, int64_t ldx,
    int64_t* K )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int forwrd_ = (lapack_int) forwrd;
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int ldx_ = (lapack_int) ldx;
    #if 1
        // 32-bit copy
        std::vector< lapack_int > K_( &K[0], &K[(m)] );
        lapack_int* K_ptr = &K_[0];
    #else
        lapack_int* K_ptr = K;
    #endif

    LAPACK_slapmr(
        &forwrd_, &m_, &n_,
        X, &ldx_,
        K_ptr );
    #if 1
        std::copy( K_.begin(), K_.end(), K );
    #endif
}

// -----------------------------------------------------------------------------
void lapmr(
    bool forwrd, int64_t m, int64_t n,
    double* X, int64_t ldx,
    int64_t* K )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int forwrd_ = (lapack_int) forwrd;
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int ldx_ = (lapack_int) ldx;
    #if 1
        // 32-bit copy
        std::vector< lapack_int > K_( &K[0], &K[(m)] );
        lapack_int* K_ptr = &K_[0];
    #else
        lapack_int* K_ptr = K;
    #endif

    LAPACK_dlapmr(
        &forwrd_, &m_, &n_,
        X, &ldx_,
        K_ptr );
    #if 1
        std::copy( K_.begin(), K_.end(), K );
    #endif
}

// -----------------------------------------------------------------------------
void lapmr(
    bool forwrd, int64_t m, int64_t n,
    std::complex<float>* X, int64_t ldx,
    int64_t* K )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int forwrd_ = (lapack_int) forwrd;
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int ldx_ = (lapack_int) ldx;
    #if 1
        // 32-bit copy
        std::vector< lapack_int > K_( &K[0], &K[(m)] );
        lapack_int* K_ptr = &K_[0];
    #else
        lapack_int* K_ptr = K;
    #endif

    LAPACK_clapmr(
        &forwrd_, &m_, &n_,
        (lapack_complex_float*) X, &ldx_,
        K_ptr );
    #if 1
        std::copy( K_.begin(), K_.end(), K );
    #endif
}

// -----------------------------------------------------------------------------
void lapmr(
    bool forwrd, int64_t m, int64_t n,
    std::complex<double>* X, int64_t ldx,
    int64_t* K )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int forwrd_ = (lapack_int) forwrd;
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int ldx_ = (lapack_int) ldx;
    #if 1
        // 32-bit copy
        std::vector< lapack_int > K_( &K[0], &K[(m)] );
        lapack_int* K_ptr = &K_[0];
    #else
        lapack_int* K_ptr = K;
    #endif

    LAPACK_zlapmr(
        &forwrd_, &m_, &n_,
        (lapack_complex_double*) X, &ldx_,
        K_ptr );
    #if 1
        std::copy( K_.begin(), K_.end(), K );
    #endif
}

}  // namespace lapack

#endif  // LAPACK >= v3.3
