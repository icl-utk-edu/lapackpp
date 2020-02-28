#include "lapack.hh"
#include "lapack/fortran.h"

#if LAPACK_VERSION >= 30300  // >= 3.3

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup sysv_computational
void syswapr(
    lapack::Uplo uplo, int64_t n,
    float* A, int64_t lda, int64_t i1, int64_t i2 )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(i1) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(i2) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int i1_ = (lapack_int) i1;
    lapack_int i2_ = (lapack_int) i2;

    LAPACK_ssyswapr(
        &uplo_, &n_,
        A, &lda_, &i1_, &i2_ );
}

// -----------------------------------------------------------------------------
/// @ingroup sysv_computational
void syswapr(
    lapack::Uplo uplo, int64_t n,
    double* A, int64_t lda, int64_t i1, int64_t i2 )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(i1) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(i2) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int i1_ = (lapack_int) i1;
    lapack_int i2_ = (lapack_int) i2;

    LAPACK_dsyswapr(
        &uplo_, &n_,
        A, &lda_, &i1_, &i2_ );
}

// -----------------------------------------------------------------------------
/// @ingroup sysv_computational
void syswapr(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* A, int64_t lda, int64_t i1, int64_t i2 )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(i1) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(i2) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int i1_ = (lapack_int) i1;
    lapack_int i2_ = (lapack_int) i2;

    LAPACK_csyswapr(
        &uplo_, &n_,
        (lapack_complex_float*) A, &lda_, &i1_, &i2_ );
}

// -----------------------------------------------------------------------------
/// Applies an elementary permutation on the rows and the columns of
/// a symmetric matrix.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, `lapack::heswapr` is an alias for this.
/// For complex Hermitian matrices, see `lapack::heswapr`.
///
/// @param[in] uplo
///     Whether the details of the factorization are stored
///     as an upper or lower triangular matrix.
///     - lapack::Uplo::Upper: Upper triangular, form is \f$ A = U D U^T; \f$
///     - lapack::Uplo::Lower: Lower triangular, form is \f$ A = L D L^T. \f$
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     TODO (the LAPACK documentation seems wrong).
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[in] i1
///     Index of the first row to swap
///
/// @param[in] i2
///     Index of the second row to swap
///
/// @ingroup sysv_computational
void syswapr(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* A, int64_t lda, int64_t i1, int64_t i2 )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(i1) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(i2) > std::numeric_limits<lapack_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int i1_ = (lapack_int) i1;
    lapack_int i2_ = (lapack_int) i2;

    LAPACK_zsyswapr(
        &uplo_, &n_,
        (lapack_complex_double*) A, &lda_, &i1_, &i2_ );
}

}  // namespace lapack

#endif  // LAPACK >= 3.3.0
