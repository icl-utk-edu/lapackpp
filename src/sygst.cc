#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
int64_t sygst(
    int64_t itype, lapack::Uplo uplo, int64_t n,
    float* A, int64_t lda,
    float const* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(itype) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int itype_ = (lapack_int) itype;
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;
    const unsigned string_len = 1;

    LAPACK_ssygst(
        &itype_, &uplo_, &n_,
        A, &lda_,
        B, &ldb_, &info_,
        string_len );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
int64_t sygst(
    int64_t itype, lapack::Uplo uplo, int64_t n,
    double* A, int64_t lda,
    double const* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(itype) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<lapack_int>::max() );
    }
    lapack_int itype_ = (lapack_int) itype;
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    lapack_int info_ = 0;
    const unsigned string_len = 1;

    LAPACK_dsygst(
        &itype_, &uplo_, &n_,
        A, &lda_,
        B, &ldb_, &info_,
        string_len );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
