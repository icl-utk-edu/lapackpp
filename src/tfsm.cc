#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
void tfsm(
    lapack::Op transr, lapack::Side side, lapack::Uplo uplo, lapack::Op trans, lapack::Diag diag, int64_t m, int64_t n, float alpha,
    float const* A,
    float* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char transr_ = op2char( transr );
    char side_ = side2char( side );
    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_ = diag2char( diag );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ldb_ = (blas_int) ldb;

    LAPACK_stfsm( &transr_, &side_, &uplo_, &trans_, &diag_, &m_, &n_, &alpha, A, B, &ldb_ );
}

// -----------------------------------------------------------------------------
void tfsm(
    lapack::Op transr, lapack::Side side, lapack::Uplo uplo, lapack::Op trans, lapack::Diag diag, int64_t m, int64_t n, double alpha,
    double const* A,
    double* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char transr_ = op2char( transr );
    char side_ = side2char( side );
    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_ = diag2char( diag );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ldb_ = (blas_int) ldb;

    LAPACK_dtfsm( &transr_, &side_, &uplo_, &trans_, &diag_, &m_, &n_, &alpha, A, B, &ldb_ );
}

// -----------------------------------------------------------------------------
void tfsm(
    lapack::Op transr, lapack::Side side, lapack::Uplo uplo, lapack::Op trans, lapack::Diag diag, int64_t m, int64_t n, std::complex<float> alpha,
    std::complex<float> const* A,
    std::complex<float>* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char transr_ = op2char( transr );
    char side_ = side2char( side );
    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_ = diag2char( diag );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ldb_ = (blas_int) ldb;

    LAPACK_ctfsm( &transr_, &side_, &uplo_, &trans_, &diag_, &m_, &n_, &alpha, A, B, &ldb_ );
}

// -----------------------------------------------------------------------------
void tfsm(
    lapack::Op transr, lapack::Side side, lapack::Uplo uplo, lapack::Op trans, lapack::Diag diag, int64_t m, int64_t n, std::complex<double> alpha,
    std::complex<double> const* A,
    std::complex<double>* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char transr_ = op2char( transr );
    char side_ = side2char( side );
    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_ = diag2char( diag );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int ldb_ = (blas_int) ldb;

    LAPACK_ztfsm( &transr_, &side_, &uplo_, &trans_, &diag_, &m_, &n_, &alpha, A, B, &ldb_ );
}

}  // namespace lapack
