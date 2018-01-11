#include "lapack.hh"
#include "lapack_fortran.h"

// while [cz]syr are in LAPACK, [sd]syr are in BLAS,
// so we put them all in the blas namespace
namespace blas {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup syr
void syr(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *x, int64_t incx,
    std::complex<float>       *A, int64_t lda )
{
    // check arguments
    lapack_error_if( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    lapack_error_if( uplo != Uplo::Lower &&
               uplo != Uplo::Upper );
    lapack_error_if( n < 0 );
    lapack_error_if( lda < n );
    lapack_error_if( incx == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( n              > std::numeric_limits<blas_int>::max() );
        lapack_error_if( lda            > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    char uplo_ = uplo2char( uplo );
    LAPACK_csyr( &uplo_, &n_, &alpha, x, &incx_, A, &lda_ );
}

// -----------------------------------------------------------------------------
/// @ingroup syr
void syr(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *x, int64_t incx,
    std::complex<double>       *A, int64_t lda )
{
    // check arguments
    lapack_error_if( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    lapack_error_if( uplo != Uplo::Lower &&
               uplo != Uplo::Upper );
    lapack_error_if( n < 0 );
    lapack_error_if( lda < n );
    lapack_error_if( incx == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( n              > std::numeric_limits<blas_int>::max() );
        lapack_error_if( lda            > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    char uplo_ = uplo2char( uplo );
    LAPACK_zsyr( &uplo_, &n_, &alpha, x, &incx_, A, &lda_ );
}

}  // namespace blas
