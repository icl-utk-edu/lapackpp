#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
float lansy(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    float const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;

    // allocate workspace
    std::vector< float > work( max( (int64_t) 1, lwork) );

    return LAPACK_slansy( &norm_, &uplo_, &n_, A, &lda_, &work[0] );
}

// -----------------------------------------------------------------------------
double lansy(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    double const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;

    // allocate workspace
    std::vector< double > work( max( (int64_t) 1, lwork) );

    return LAPACK_dlansy( &norm_, &uplo_, &n_, A, &lda_, &work[0] );
}

// -----------------------------------------------------------------------------
float lansy(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    std::complex<float> const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;

    // allocate workspace
    std::vector< float > work( max( (int64_t) 1, lwork) );

    return LAPACK_clansy( &norm_, &uplo_, &n_, A, &lda_, &work[0] );
}

// -----------------------------------------------------------------------------
double lansy(
    lapack::Norm norm, lapack::Uplo uplo, int64_t n,
    std::complex<double> const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;

    // allocate workspace
    std::vector< double > work( max( (int64_t) 1, lwork) );

    return LAPACK_zlansy( &norm_, &uplo_, &n_, A, &lda_, &work[0] );
}

}  // namespace lapack
