#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
void larfx(
    lapack::Side side, int64_t m, int64_t n,
    float const* V, float tau,
    float* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ldc_ = (blas_int) ldc;

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    std::vector< float > work( lwork );

    LAPACK_slarfx( &side_, &m_, &n_, V, &tau, C, &ldc_, &work[0] );
}

// -----------------------------------------------------------------------------
void larfx(
    lapack::Side side, int64_t m, int64_t n,
    double const* V, double tau,
    double* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ldc_ = (blas_int) ldc;

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    std::vector< double > work( lwork );

    LAPACK_dlarfx( &side_, &m_, &n_, V, &tau, C, &ldc_, &work[0] );
}

// -----------------------------------------------------------------------------
void larfx(
    lapack::Side side, int64_t m, int64_t n,
    std::complex<float> const* V, std::complex<float> tau,
    std::complex<float>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ldc_ = (blas_int) ldc;

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork );

    LAPACK_clarfx( &side_, &m_, &n_, V, &tau, C, &ldc_, &work[0] );
}

// -----------------------------------------------------------------------------
void larfx(
    lapack::Side side, int64_t m, int64_t n,
    std::complex<double> const* V, std::complex<double> tau,
    std::complex<double>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ldc_ = (blas_int) ldc;

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork );

    LAPACK_zlarfx( &side_, &m_, &n_, V, &tau, C, &ldc_, &work[0] );
}

}  // namespace lapack
