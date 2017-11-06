#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t opmtr(
    lapack::Side side, lapack::Uplo uplo, lapack::Op trans, int64_t m, int64_t n,
    float const* AP,
    float const* TAU,
    float* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( n if side = 'l'; m if side = 'r' );

    LAPACK_sopmtr( &side_, &uplo_, &trans_, &m_, &n_, AP, TAU, C, &ldc_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t opmtr(
    lapack::Side side, lapack::Uplo uplo, lapack::Op trans, int64_t m, int64_t n,
    double const* AP,
    double const* TAU,
    double* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( n if side = 'l'; m if side = 'r' );

    LAPACK_dopmtr( &side_, &uplo_, &trans_, &m_, &n_, AP, TAU, C, &ldc_, &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
