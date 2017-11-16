#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
void sfrk(
    lapack::Op transr, lapack::Uplo uplo, lapack::Op trans, int64_t n, int64_t k, float alpha,
    float const* A, int64_t lda, float beta,
    float* C )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char transr_ = op2char( transr );
    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int lda_ = (blas_int) lda;

    LAPACK_ssfrk( &transr_, &uplo_, &trans_, &n_, &k_, &alpha, A, &lda_, &beta, C );
}

// -----------------------------------------------------------------------------
void sfrk(
    lapack::Op transr, lapack::Uplo uplo, lapack::Op trans, int64_t n, int64_t k, double alpha,
    double const* A, int64_t lda, double beta,
    double* C )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(k) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char transr_ = op2char( transr );
    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int lda_ = (blas_int) lda;

    LAPACK_dsfrk( &transr_, &uplo_, &trans_, &n_, &k_, &alpha, A, &lda_, &beta, C );
}

}  // namespace lapack
