#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t stein(
    int64_t n,
    float const* D,
    float const* E, int64_t m,
    float const* W,
    int64_t const* iblock,
    int64_t const* isplit,
    float* Z, int64_t ldz,
    int64_t* ifail )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int m_ = (blas_int) m;
    #if 1
        // 32-bit copy
        std::vector< blas_int > iblock_( &iblock[0], &iblock[(n)] );
        blas_int const* iblock_ptr = &iblock_[0];
    #else
        blas_int const* iblock_ptr = iblock_;
    #endif
    #if 1
        // 32-bit copy
        std::vector< blas_int > isplit_( &isplit[0], &isplit[(n)] );
        blas_int const* isplit_ptr = &isplit_[0];
    #else
        blas_int const* isplit_ptr = isplit_;
    #endif
    blas_int ldz_ = (blas_int) ldz;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ifail_( (m) );
        blas_int* ifail_ptr = &ifail_[0];
    #else
        blas_int* ifail_ptr = ifail;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (5*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_sstein( &n_, D, E, &m_, W, iblock_ptr, isplit_ptr, Z, &ldz_, &work[0], &iwork[0], ifail_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( ifail_.begin(), ifail_.end(), ifail );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t stein(
    int64_t n,
    double const* D,
    double const* E, int64_t m,
    double const* W,
    int64_t const* iblock,
    int64_t const* isplit,
    double* Z, int64_t ldz,
    int64_t* ifail )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int m_ = (blas_int) m;
    #if 1
        // 32-bit copy
        std::vector< blas_int > iblock_( &iblock[0], &iblock[(n)] );
        blas_int const* iblock_ptr = &iblock_[0];
    #else
        blas_int const* iblock_ptr = iblock_;
    #endif
    #if 1
        // 32-bit copy
        std::vector< blas_int > isplit_( &isplit[0], &isplit[(n)] );
        blas_int const* isplit_ptr = &isplit_[0];
    #else
        blas_int const* isplit_ptr = isplit_;
    #endif
    blas_int ldz_ = (blas_int) ldz;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ifail_( (m) );
        blas_int* ifail_ptr = &ifail_[0];
    #else
        blas_int* ifail_ptr = ifail;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (5*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_dstein( &n_, D, E, &m_, W, iblock_ptr, isplit_ptr, Z, &ldz_, &work[0], &iwork[0], ifail_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( ifail_.begin(), ifail_.end(), ifail );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t stein(
    int64_t n,
    float const* D,
    float const* E, int64_t m,
    float const* W,
    int64_t const* iblock,
    int64_t const* isplit,
    std::complex<float>* Z, int64_t ldz,
    int64_t* ifail )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int m_ = (blas_int) m;
    #if 1
        // 32-bit copy
        std::vector< blas_int > iblock_( &iblock[0], &iblock[(n)] );
        blas_int const* iblock_ptr = &iblock_[0];
    #else
        blas_int const* iblock_ptr = iblock_;
    #endif
    #if 1
        // 32-bit copy
        std::vector< blas_int > isplit_( &isplit[0], &isplit[(n)] );
        blas_int const* isplit_ptr = &isplit_[0];
    #else
        blas_int const* isplit_ptr = isplit_;
    #endif
    blas_int ldz_ = (blas_int) ldz;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ifail_( (m) );
        blas_int* ifail_ptr = &ifail_[0];
    #else
        blas_int* ifail_ptr = ifail;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (5*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_cstein( &n_, D, E, &m_, W, iblock_ptr, isplit_ptr, Z, &ldz_, &work[0], &iwork[0], ifail_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( ifail_.begin(), ifail_.end(), ifail );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
int64_t stein(
    int64_t n,
    double const* D,
    double const* E, int64_t m,
    double const* W,
    int64_t const* iblock,
    int64_t const* isplit,
    std::complex<double>* Z, int64_t ldz,
    int64_t* ifail )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldz) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int m_ = (blas_int) m;
    #if 1
        // 32-bit copy
        std::vector< blas_int > iblock_( &iblock[0], &iblock[(n)] );
        blas_int const* iblock_ptr = &iblock_[0];
    #else
        blas_int const* iblock_ptr = iblock_;
    #endif
    #if 1
        // 32-bit copy
        std::vector< blas_int > isplit_( &isplit[0], &isplit[(n)] );
        blas_int const* isplit_ptr = &isplit_[0];
    #else
        blas_int const* isplit_ptr = isplit_;
    #endif
    blas_int ldz_ = (blas_int) ldz;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ifail_( (m) );
        blas_int* ifail_ptr = &ifail_[0];
    #else
        blas_int* ifail_ptr = ifail;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (5*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_zstein( &n_, D, E, &m_, W, iblock_ptr, isplit_ptr, Z, &ldz_, &work[0], &iwork[0], ifail_ptr, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( ifail_.begin(), ifail_.end(), ifail );
    #endif
    return info_;
}

}  // namespace lapack
