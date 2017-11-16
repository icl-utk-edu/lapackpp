#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t gttrs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    float const* DL,
    float const* D,
    float const* DU,
    float const* DU2,
    int64_t const* ipiv,
    float* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    LAPACK_sgttrs( &trans_, &n_, &nrhs_, DL, D, DU, DU2, ipiv_ptr, B, &ldb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gttrs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    double const* DL,
    double const* D,
    double const* DU,
    double const* DU2,
    int64_t const* ipiv,
    double* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    LAPACK_dgttrs( &trans_, &n_, &nrhs_, DL, D, DU, DU2, ipiv_ptr, B, &ldb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gttrs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    std::complex<float> const* DL,
    std::complex<float> const* D,
    std::complex<float> const* DU,
    std::complex<float> const* DU2,
    int64_t const* ipiv,
    std::complex<float>* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    LAPACK_cgttrs( &trans_, &n_, &nrhs_, DL, D, DU, DU2, ipiv_ptr, B, &ldb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t gttrs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    std::complex<double> const* DL,
    std::complex<double> const* D,
    std::complex<double> const* DU,
    std::complex<double> const* DU2,
    int64_t const* ipiv,
    std::complex<double>* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    LAPACK_zgttrs( &trans_, &n_, &nrhs_, DL, D, DU, DU2, ipiv_ptr, B, &ldb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
