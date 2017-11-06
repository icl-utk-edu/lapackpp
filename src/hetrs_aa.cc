#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t hetrs_aa(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<float> const* A, int64_t lda,
    int64_t const* ipiv,
    std::complex<float>* B, int64_t ldb, int64_t lwork )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lwork) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int lwork_ = (blas_int) lwork;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (max( (int64_t) 1, lwork)) );

    LAPACK_chetrs_aa( &uplo_, &n_, &nrhs_, A, &lda_, ipiv_ptr, B, &ldb_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t hetrs_aa(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<double> const* A, int64_t lda,
    int64_t const* ipiv,
    std::complex<double>* B, int64_t ldb, int64_t lwork )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lwork) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int lwork_ = (blas_int) lwork;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (max( (int64_t) 1, lwork)) );

    LAPACK_zhetrs_aa( &uplo_, &n_, &nrhs_, A, &lda_, ipiv_ptr, B, &ldb_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
