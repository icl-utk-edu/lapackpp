#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t ggbak(
    lapack::Balance balance, lapack::Side side, int64_t n, int64_t ilo, int64_t ihi,
    float const* lscale,
    float const* rscale, int64_t m,
    float* V, int64_t ldv )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    char side_ = side2char( side );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int m_ = (blas_int) m;
    blas_int ldv_ = (blas_int) ldv;
    blas_int info_ = 0;

    LAPACK_sggbak(
        &balance_, &side_, &n_, &ilo_, &ihi_,
        lscale,
        rscale, &m_,
        V, &ldv_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbak(
    lapack::Balance balance, lapack::Side side, int64_t n, int64_t ilo, int64_t ihi,
    double const* lscale,
    double const* rscale, int64_t m,
    double* V, int64_t ldv )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    char side_ = side2char( side );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int m_ = (blas_int) m;
    blas_int ldv_ = (blas_int) ldv;
    blas_int info_ = 0;

    LAPACK_dggbak(
        &balance_, &side_, &n_, &ilo_, &ihi_,
        lscale,
        rscale, &m_,
        V, &ldv_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbak(
    lapack::Balance balance, lapack::Side side, int64_t n, int64_t ilo, int64_t ihi,
    float const* lscale,
    float const* rscale, int64_t m,
    std::complex<float>* V, int64_t ldv )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    char side_ = side2char( side );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int m_ = (blas_int) m;
    blas_int ldv_ = (blas_int) ldv;
    blas_int info_ = 0;

    LAPACK_cggbak(
        &balance_, &side_, &n_, &ilo_, &ihi_,
        lscale,
        rscale, &m_,
        (lapack_complex_float*) V, &ldv_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbak(
    lapack::Balance balance, lapack::Side side, int64_t n, int64_t ilo, int64_t ihi,
    double const* lscale,
    double const* rscale, int64_t m,
    std::complex<double>* V, int64_t ldv )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
    }
    char balance_ = balance2char( balance );
    char side_ = side2char( side );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int m_ = (blas_int) m;
    blas_int ldv_ = (blas_int) ldv;
    blas_int info_ = 0;

    LAPACK_zggbak(
        &balance_, &side_, &n_, &ilo_, &ihi_,
        lscale,
        rscale, &m_,
        (lapack_complex_double*) V, &ldv_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
