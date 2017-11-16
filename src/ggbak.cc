#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t ggbak(
    lapack::Job job, lapack::Side side, int64_t n, int64_t ilo, int64_t ihi,
    float const* LSCALE,
    float const* RSCALE, int64_t m,
    float* V, int64_t ldv )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    char side_ = side2char( side );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int m_ = (blas_int) m;
    blas_int ldv_ = (blas_int) ldv;
    blas_int info_ = 0;

    LAPACK_sggbak( &job_, &side_, &n_, &ilo_, &ihi_, LSCALE, RSCALE, &m_, V, &ldv_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbak(
    lapack::Job job, lapack::Side side, int64_t n, int64_t ilo, int64_t ihi,
    double const* LSCALE,
    double const* RSCALE, int64_t m,
    double* V, int64_t ldv )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    char side_ = side2char( side );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int m_ = (blas_int) m;
    blas_int ldv_ = (blas_int) ldv;
    blas_int info_ = 0;

    LAPACK_dggbak( &job_, &side_, &n_, &ilo_, &ihi_, LSCALE, RSCALE, &m_, V, &ldv_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbak(
    lapack::Job job, lapack::Side side, int64_t n, int64_t ilo, int64_t ihi,
    float const* LSCALE,
    float const* RSCALE, int64_t m,
    std::complex<float>* V, int64_t ldv )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    char side_ = side2char( side );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int m_ = (blas_int) m;
    blas_int ldv_ = (blas_int) ldv;
    blas_int info_ = 0;

    LAPACK_cggbak( &job_, &side_, &n_, &ilo_, &ihi_, LSCALE, RSCALE, &m_, V, &ldv_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t ggbak(
    lapack::Job job, lapack::Side side, int64_t n, int64_t ilo, int64_t ihi,
    double const* LSCALE,
    double const* RSCALE, int64_t m,
    std::complex<double>* V, int64_t ldv )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ilo) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ihi) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldv) > std::numeric_limits<blas_int>::max() );
    }
    char job_ = job2char( job );
    char side_ = side2char( side );
    blas_int n_ = (blas_int) n;
    blas_int ilo_ = (blas_int) ilo;
    blas_int ihi_ = (blas_int) ihi;
    blas_int m_ = (blas_int) m;
    blas_int ldv_ = (blas_int) ldv;
    blas_int info_ = 0;

    LAPACK_zggbak( &job_, &side_, &n_, &ilo_, &ihi_, LSCALE, RSCALE, &m_, V, &ldv_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
