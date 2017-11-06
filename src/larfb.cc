#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
void larfb(
    lapack::Side side, lapack::Op trans, lapack::Direct direct, lapack::StoreV storev, int64_t m, int64_t n, int64_t k,
    float const* V, int64_t ldv,
    float const* T, int64_t ldt,
    float* C, int64_t ldc, int64_t ldwork )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldwork) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    char direct_ = direct2char( direct );
    char storev_ = storev2char( storev );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldc_ = (blas_int) ldc;
    blas_int ldwork_ = (blas_int) ldwork;

    // allocate workspace
    std::vector< float > work( (ldwork,k) );

    LAPACK_slarfb( &side_, &trans_, &direct_, &storev_, &m_, &n_, &k_, V, &ldv_, T, &ldt_, C, &ldc_, &work[0], &ldwork_ );
}

// -----------------------------------------------------------------------------
void larfb(
    lapack::Side side, lapack::Op trans, lapack::Direct direct, lapack::StoreV storev, int64_t m, int64_t n, int64_t k,
    double const* V, int64_t ldv,
    double const* T, int64_t ldt,
    double* C, int64_t ldc, int64_t ldwork )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldwork) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    char direct_ = direct2char( direct );
    char storev_ = storev2char( storev );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldc_ = (blas_int) ldc;
    blas_int ldwork_ = (blas_int) ldwork;

    // allocate workspace
    std::vector< double > work( (ldwork,k) );

    LAPACK_dlarfb( &side_, &trans_, &direct_, &storev_, &m_, &n_, &k_, V, &ldv_, T, &ldt_, C, &ldc_, &work[0], &ldwork_ );
}

// -----------------------------------------------------------------------------
void larfb(
    lapack::Side side, lapack::Op trans, lapack::Direct direct, lapack::StoreV storev, int64_t m, int64_t n, int64_t k,
    std::complex<float> const* V, int64_t ldv,
    std::complex<float> const* T, int64_t ldt,
    std::complex<float>* C, int64_t ldc, int64_t ldwork )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldwork) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    char direct_ = direct2char( direct );
    char storev_ = storev2char( storev );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldc_ = (blas_int) ldc;
    blas_int ldwork_ = (blas_int) ldwork;

    // allocate workspace
    std::vector< std::complex<float> > work( (ldwork,k) );

    LAPACK_clarfb( &side_, &trans_, &direct_, &storev_, &m_, &n_, &k_, V, &ldv_, T, &ldt_, C, &ldc_, &work[0], &ldwork_ );
}

// -----------------------------------------------------------------------------
void larfb(
    lapack::Side side, lapack::Op trans, lapack::Direct direct, lapack::StoreV storev, int64_t m, int64_t n, int64_t k,
    std::complex<double> const* V, int64_t ldv,
    std::complex<double> const* T, int64_t ldt,
    std::complex<double>* C, int64_t ldc, int64_t ldwork )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldt) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldwork) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    char direct_ = direct2char( direct );
    char storev_ = storev2char( storev );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int ldv_ = (blas_int) ldv;
    blas_int ldt_ = (blas_int) ldt;
    blas_int ldc_ = (blas_int) ldc;
    blas_int ldwork_ = (blas_int) ldwork;

    // allocate workspace
    std::vector< std::complex<double> > work( (ldwork,k) );

    LAPACK_zlarfb( &side_, &trans_, &direct_, &storev_, &m_, &n_, &k_, V, &ldv_, T, &ldt_, C, &ldc_, &work[0], &ldwork_ );
}

}  // namespace lapack
