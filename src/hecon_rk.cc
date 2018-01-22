#include "lapack.hh"
#include "lapack_fortran.h"

#if LAPACK_VERSION >= 30700  // >= 3.7

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup hesv_rk_computational
int64_t hecon_rk(
    lapack::Uplo uplo, int64_t n,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* E,
    int64_t const* ipiv, float anorm,
    float* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (2*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_checon_3( &uplo_, &n_, A, &lda_, E, ipiv_ptr, &anorm, rcond, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Estimates the reciprocal of the condition number (in the
/// 1-norm) of a Hermitian matrix A using the factorization
/// computed by `lapack::hetrf_rk`:
///
///     \f$ A = P U D U^H P^T \f$ or
///     \f$ A = P L D L^H P^T, \f$
///
/// where U (or L) is unit upper (or lower) triangular matrix,
/// \f$ U^H \f$ (or \f$ L^H \f$) is the conjugate of U (or L), P is a permutation
/// matrix, \f$ P^T \f$ is the transpose of P, and D is Hermitian and block
/// diagonal with 1-by-1 and 2-by-2 diagonal blocks.
///
/// An estimate is obtained for \f$ || A^{-1} ||_1, \f$ and the reciprocal of the
/// condition number is computed as \f$ \text{rcond} = 1 / (|| A ||_1 * || A^{-1} ||_1). \f$
/// This routine uses the BLAS-3 solver `lapack::hetrs_rk`.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this is an alias for `lapack::sycon_rk`.
/// For complex symmetric matrices, see `lapack::sycon_rk`.
///
/// @since LAPACK 3.7.0.
///
/// @param[in] uplo
///     Specifies whether the details of the factorization are
///     stored as an upper or lower triangular matrix:
///     - lapack::Uplo::Upper: Upper triangular, form is \f$ A = P U D U^H P^T; \f$
///     - lapack::Uplo::Lower: Lower triangular, form is \f$ A = P L D L^H P^T. \f$
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     Diagonal of the block diagonal matrix D and factors U or L
///     as computed by `lapack::hetrf_rk`:
///     - ONLY diagonal elements of the Hermitian block diagonal
///         matrix D on the diagonal of A, i.e. D(k,k) = A(k,k);
///         (superdiagonal (or subdiagonal) elements of D
///         should be provided on entry in array E), and
///     - If uplo = Upper: factor U in the superdiagonal part of A.
///     - If uplo = Lower: factor L in the subdiagonal part of A.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[in] E
///     The vector E of length n.
///     On entry, contains the superdiagonal (or subdiagonal)
///     elements of the Hermitian block diagonal matrix D
///     with 1-by-1 or 2-by-2 diagonal blocks, where
///     - If uplo = Upper: E(i) = D(i-1,i),i=2:n, E(1) not referenced;
///     - If uplo = Lower: E(i) = D(i+1,i),i=1:n-1, E(n) not referenced.
///
///     - Note: For 1-by-1 diagonal block D(k), where
///     1 <= k <= n, the element E(k) is not referenced in both
///     uplo = Upper or uplo = Lower cases.
///
/// @param[in] ipiv
///     The vector ipiv of length n.
///     Details of the interchanges and the block structure of D
///     as determined by `lapack::hetrf_rk`.
///
/// @param[in] anorm
///     The 1-norm of the original matrix A.
///
/// @param[out] rcond
///     The reciprocal of the condition number of the matrix A,
///     computed as rcond = 1/(anorm * ainv_norm), where ainv_norm is an
///     estimate of the 1-norm of \f$ A^{-1} \f$ computed in this routine.
///
/// @retval = 0: successful exit
///
/// @ingroup hesv_rk_computational
int64_t hecon_rk(
    lapack::Uplo uplo, int64_t n,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* E,
    int64_t const* ipiv, double anorm,
    double* rcond )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (2*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_zhecon_3( &uplo_, &n_, A, &lda_, E, ipiv_ptr, &anorm, rcond, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack

#endif  // LAPACK >= 3.7
