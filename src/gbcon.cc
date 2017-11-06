#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t gbcon(
    lapack::Norm norm, int64_t n, int64_t kl, int64_t ku,
    float const* AB, int64_t ldab,
    int64_t const* ipiv, float anorm,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    blas_int n_ = (blas_int) n;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int ldab_ = (blas_int) ldab;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (3*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_sgbcon( &norm_, &n_, &kl_, &ku_, AB, &ldab_, ipiv_ptr, &anorm, rcond, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gbcon(
    lapack::Norm norm, int64_t n, int64_t kl, int64_t ku,
    double const* AB, int64_t ldab,
    int64_t const* ipiv, double anorm,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    blas_int n_ = (blas_int) n;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int ldab_ = (blas_int) ldab;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (3*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_dgbcon( &norm_, &n_, &kl_, &ku_, AB, &ldab_, ipiv_ptr, &anorm, rcond, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gbcon(
    lapack::Norm norm, int64_t n, int64_t kl, int64_t ku,
    std::complex<float> const* AB, int64_t ldab,
    int64_t const* ipiv, float anorm,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    blas_int n_ = (blas_int) n;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int ldab_ = (blas_int) ldab;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (2*n) );
    std::vector< float > rwork( (n) );

    LAPACK_cgbcon( &norm_, &n_, &kl_, &ku_, AB, &ldab_, ipiv_ptr, &anorm, rcond, &work[0], &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gbcon(
    lapack::Norm norm, int64_t n, int64_t kl, int64_t ku,
    std::complex<double> const* AB, int64_t ldab,
    int64_t const* ipiv, double anorm,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldab) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    blas_int n_ = (blas_int) n;
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int ldab_ = (blas_int) ldab;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (2*n) );
    std::vector< double > rwork( (n) );

    LAPACK_zgbcon( &norm_, &n_, &kl_, &ku_, AB, &ldab_, ipiv_ptr, &anorm, rcond, &work[0], &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
