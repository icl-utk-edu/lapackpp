#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gerqf
int64_t ormrq(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t k,
    float const* A, int64_t lda,
    float const* tau,
    float* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_sormrq( &side_, &trans_, &m_, &n_, &k_, A, &lda_, tau, C, &ldc_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_sormrq( &side_, &trans_, &m_, &n_, &k_, A, &lda_, tau, C, &ldc_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @see lapack::unmrq
/// @ingroup gerqf
int64_t ormrq(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t k,
    double const* A, int64_t lda,
    double const* tau,
    double* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dormrq( &side_, &trans_, &m_, &n_, &k_, A, &lda_, tau, C, &ldc_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dormrq( &side_, &trans_, &m_, &n_, &k_, A, &lda_, tau, C, &ldc_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
