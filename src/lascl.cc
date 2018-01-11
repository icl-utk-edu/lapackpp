#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup auxiliary
int64_t lascl(
    lapack::MatrixType type, int64_t kl, int64_t ku, float cfrom, float cto, int64_t m, int64_t n,
    float* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char type_ = matrixtype2char( type );
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    LAPACK_slascl( &type_, &kl_, &ku_, &cfrom, &cto, &m_, &n_, A, &lda_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup auxiliary
int64_t lascl(
    lapack::MatrixType type, int64_t kl, int64_t ku, double cfrom, double cto, int64_t m, int64_t n,
    double* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char type_ = matrixtype2char( type );
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    LAPACK_dlascl( &type_, &kl_, &ku_, &cfrom, &cto, &m_, &n_, A, &lda_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup auxiliary
int64_t lascl(
    lapack::MatrixType type, int64_t kl, int64_t ku, float cfrom, float cto, int64_t m, int64_t n,
    std::complex<float>* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char type_ = matrixtype2char( type );
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    LAPACK_clascl( &type_, &kl_, &ku_, &cfrom, &cto, &m_, &n_, A, &lda_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Multiplies the m-by-n complex matrix A by the real scalar
/// cto / cfrom. This is done without over/underflow as long as the final
/// result cto * A(i,j) / cfrom does not over/underflow. type specifies that
/// A may be full, upper triangular, lower triangular, upper Hessenberg,
/// or banded.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] type
///     type indices the storage type of the input matrix.
///     - lapack::MatrixType::General:
///         A is a full matrix.
///
///     - lapack::MatrixType::Lower:
///         A is a lower triangular matrix.
///
///     - lapack::MatrixType::Upper:
///         A is an upper triangular matrix.
///
///     - lapack::MatrixType::Hessenberg:
///         A is an upper Hessenberg matrix.
///
///     - lapack::MatrixType::LowerBand:
///         A is a symmetric band matrix with lower bandwidth kl
///         and upper bandwidth ku and with the only the lower
///         half stored.
///
///     - lapack::MatrixType::UpperBand:
///         A is a symmetric band matrix with lower bandwidth kl
///         and upper bandwidth ku and with the only the upper
///         half stored.
///
///     - lapack::MatrixType::Band:
///         A is a band matrix with lower bandwidth kl and upper
///         bandwidth ku. See `lapack::gbtrf` for storage details.
///
/// @param[in] kl
///     The lower bandwidth of A.
///     Referenced only if type = LowerBand, UpperBand, or Band.
///
/// @param[in] ku
///     The upper bandwidth of A.
///     Referenced only if type = LowerBand, UpperBand, or Band.
///
/// @param[in] cfrom
///
/// @param[in] cto
///     The matrix A is multiplied by cto/cfrom. A(i,j) is computed
///     without over/underflow if the final result cto*A(i,j)/cfrom
///     can be represented without over/underflow. cfrom must be
///     nonzero.
///
/// @param[in] m
///     The number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix A. n >= 0.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     The matrix to be multiplied by cto/cfrom. See type for the
///     storage type.
///
/// @param[in] lda
///     The leading dimension of the array A.
///     - If type = General, Lower, Upper, or Hessenberg, lda >= max(1,m);
///     - if type = LowerBand, lda >= kl+1;
///     - if type = UpperBand, lda >= ku+1;
///     - if type = Band, lda >= 2*kl+ku+1.
///
/// @retval = 0: successful exit
///
/// @ingroup auxiliary
int64_t lascl(
    lapack::MatrixType type, int64_t kl, int64_t ku, double cfrom, double cto, int64_t m, int64_t n,
    std::complex<double>* A, int64_t lda )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(kl) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ku) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char type_ = matrixtype2char( type );
    blas_int kl_ = (blas_int) kl;
    blas_int ku_ = (blas_int) ku;
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int info_ = 0;

    LAPACK_zlascl( &type_, &kl_, &ku_, &cfrom, &cto, &m_, &n_, A, &lda_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
