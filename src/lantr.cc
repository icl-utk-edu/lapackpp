#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
float lantr(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag, int64_t m, int64_t n,
    float const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf ? m : 1);

    // allocate workspace
    std::vector< float > work( max( (int64_t) 1, lwork) );

    return LAPACK_slantr( &norm_, &uplo_, &diag_, &m_, &n_, A, &lda_, &work[0] );
}

// -----------------------------------------------------------------------------
double lantr(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag, int64_t m, int64_t n,
    double const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf ? m : 1);

    // allocate workspace
    std::vector< double > work( max( (int64_t) 1, lwork) );

    return LAPACK_dlantr( &norm_, &uplo_, &diag_, &m_, &n_, A, &lda_, &work[0] );
}

// -----------------------------------------------------------------------------
float lantr(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag, int64_t m, int64_t n,
    std::complex<float> const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf ? m : 1);

    // allocate workspace
    std::vector< float > work( max( (int64_t) 1, lwork) );

    return LAPACK_clantr( &norm_, &uplo_, &diag_, &m_, &n_, A, &lda_, &work[0] );
}

// -----------------------------------------------------------------------------
double lantr(
    lapack::Norm norm, lapack::Uplo uplo, lapack::Diag diag, int64_t m, int64_t n,
    std::complex<double> const* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char norm_ = norm2char( norm );
    char uplo_ = uplo2char( uplo );
    char diag_ = diag2char( diag );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;

    // from docs
    int64_t lwork = (norm == Norm::Inf ? m : 1);

    // allocate workspace
    std::vector< double > work( max( (int64_t) 1, lwork) );

    return LAPACK_zlantr( &norm_, &uplo_, &diag_, &m_, &n_, A, &lda_, &work[0] );
}

}  // namespace lapack
