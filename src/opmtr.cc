#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t opmtr(
    lapack::Side side, lapack::Uplo uplo, lapack::Op trans, int64_t m, int64_t n,
    float const* AP,
    float const* tau,
    float* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // for real, map ConjTrans to Trans
    if (trans_ == 'C')
        trans_ = 'T';

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    std::vector< float > work( lwork );

    LAPACK_sopmtr(
        &side_, &uplo_, &trans_, &m_, &n_,
        AP,
        tau,
        C, &ldc_,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t opmtr(
    lapack::Side side, lapack::Uplo uplo, lapack::Op trans, int64_t m, int64_t n,
    double const* AP,
    double const* tau,
    double* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // for real, map ConjTrans to Trans
    if (trans_ == 'C')
        trans_ = 'T';

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    std::vector< double > work( lwork );

    LAPACK_dopmtr(
        &side_, &uplo_, &trans_, &m_, &n_,
        AP,
        tau,
        C, &ldc_,
        &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
