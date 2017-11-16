#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
void larft(
    lapack::Direct direct, lapack::StoreV storev, int64_t n, int64_t k,
    float const* V, int64_t ldv,
    float const* TAU,
    float* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
    }
    char direct_ = direct2char( direct );
    char storev_ = storev2char( storev );
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldt_ = (blas_int) ldt;

    LAPACK_slarft( &direct_, &storev_, &n_, &k_, V, &ldv_, TAU, T, &ldt_ );
}

// -----------------------------------------------------------------------------
void larft(
    lapack::Direct direct, lapack::StoreV storev, int64_t n, int64_t k,
    double const* V, int64_t ldv,
    double const* TAU,
    double* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
    }
    char direct_ = direct2char( direct );
    char storev_ = storev2char( storev );
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldt_ = (blas_int) ldt;

    LAPACK_dlarft( &direct_, &storev_, &n_, &k_, V, &ldv_, TAU, T, &ldt_ );
}

// -----------------------------------------------------------------------------
void larft(
    lapack::Direct direct, lapack::StoreV storev, int64_t n, int64_t k,
    std::complex<float> const* V, int64_t ldv,
    std::complex<float> const* TAU,
    std::complex<float>* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
    }
    char direct_ = direct2char( direct );
    char storev_ = storev2char( storev );
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldt_ = (blas_int) ldt;

    LAPACK_clarft( &direct_, &storev_, &n_, &k_, V, &ldv_, TAU, T, &ldt_ );
}

// -----------------------------------------------------------------------------
void larft(
    lapack::Direct direct, lapack::StoreV storev, int64_t n, int64_t k,
    std::complex<double> const* V, int64_t ldv,
    std::complex<double> const* TAU,
    std::complex<double>* T, int64_t ldt )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
    }
    char direct_ = direct2char( direct );
    char storev_ = storev2char( storev );
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldt_ = (blas_int) ldt;

    LAPACK_zlarft( &direct_, &storev_, &n_, &k_, V, &ldv_, TAU, T, &ldt_ );
}

}  // namespace lapack
