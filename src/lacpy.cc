#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup initialize
void lacpy(
    lapack::MatrixType matrixtype, int64_t m, int64_t n,
    float const* A, int64_t lda,
    float* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char matrixtype_ = matrixtype2char( matrixtype );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;

    LAPACK_slacpy( &matrixtype_, &m_, &n_, A, &lda_, B, &ldb_ );
}

// -----------------------------------------------------------------------------
/// @ingroup initialize
void lacpy(
    lapack::MatrixType matrixtype, int64_t m, int64_t n,
    double const* A, int64_t lda,
    double* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char matrixtype_ = matrixtype2char( matrixtype );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;

    LAPACK_dlacpy( &matrixtype_, &m_, &n_, A, &lda_, B, &ldb_ );
}

// -----------------------------------------------------------------------------
/// @ingroup initialize
void lacpy(
    lapack::MatrixType matrixtype, int64_t m, int64_t n,
    std::complex<float> const* A, int64_t lda,
    std::complex<float>* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char matrixtype_ = matrixtype2char( matrixtype );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;

    LAPACK_clacpy( &matrixtype_, &m_, &n_, A, &lda_, B, &ldb_ );
}

// -----------------------------------------------------------------------------
/// Copies all or part of a two-dimensional matrix A to another
/// matrix B.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] matrixtype
///     Specifies the part of the matrix A to be copied to B.
///     - lapack::MatrixType::Upper: Upper triangular part
///     - lapack::MatrixType::Lower: Lower triangular part
///     - lapack::MatrixType::General: All of the matrix A
///
/// @param[in] m
///     The number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix A. n >= 0.
///
/// @param[in] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     The m-by-n matrix A.
///     - If matrixtype = Upper, only the upper trapezium is accessed;
///     - if matrixtype = Lower, only the lower trapezium is accessed.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @param[out] B
///     The m-by-n matrix B, stored in an ldb-by-n array.
///     On exit, \f$ B = A \f$ in the locations specified by matrixtype.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,m).
///
/// @ingroup initialize
void lacpy(
    lapack::MatrixType matrixtype, int64_t m, int64_t n,
    std::complex<double> const* A, int64_t lda,
    std::complex<double>* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char matrixtype_ = matrixtype2char( matrixtype );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;

    LAPACK_zlacpy( &matrixtype_, &m_, &n_, A, &lda_, B, &ldb_ );
}

}  // namespace lapack
