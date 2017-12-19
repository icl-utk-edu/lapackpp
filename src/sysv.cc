#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup sysv
int64_t sysv(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    float* A, int64_t lda,
    int64_t* ipiv,
    float* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( (n) );
        blas_int* ipiv_ptr = &ipiv_[0];
    #else
        blas_int* ipiv_ptr = ipiv;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_ssysv( &uplo_, &n_, &nrhs_, A, &lda_, ipiv_ptr, B, &ldb_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_ssysv( &uplo_, &n_, &nrhs_, A, &lda_, ipiv_ptr, B, &ldb_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup sysv
int64_t sysv(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    double* A, int64_t lda,
    int64_t* ipiv,
    double* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( (n) );
        blas_int* ipiv_ptr = &ipiv_[0];
    #else
        blas_int* ipiv_ptr = ipiv;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dsysv( &uplo_, &n_, &nrhs_, A, &lda_, ipiv_ptr, B, &ldb_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dsysv( &uplo_, &n_, &nrhs_, A, &lda_, ipiv_ptr, B, &ldb_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup sysv
int64_t sysv(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<float>* A, int64_t lda,
    int64_t* ipiv,
    std::complex<float>* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( (n) );
        blas_int* ipiv_ptr = &ipiv_[0];
    #else
        blas_int* ipiv_ptr = ipiv;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_csysv( &uplo_, &n_, &nrhs_, A, &lda_, ipiv_ptr, B, &ldb_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );

    LAPACK_csysv( &uplo_, &n_, &nrhs_, A, &lda_, ipiv_ptr, B, &ldb_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

// -----------------------------------------------------------------------------
/// Computes the solution to a system of linear equations
///     \f$ A X = B \f$,
/// where A is an n-by-n symmetric matrix and X and B are n-by-nrhs
/// matrices.
///
/// The diagonal pivoting method is used to factor A as
///     \f$ A = U D U^T \f$, if uplo = Upper, or
///     \f$ A = L D L^T \f$, if uplo = Lower,
/// where U (or L) is a product of permutation and unit upper (lower)
/// triangular matrices, and D is symmetric and block diagonal with
/// 1-by-1 and 2-by-2 diagonal blocks. The factored form of A is then
/// used to solve the system of equations \f$ A X = B \f$.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, `lapack::hesv` is an alias for this.
/// For complex Hermitian matrices, @see `lapack::hesv`.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangle of A is stored;
///     - lapack::Uplo::Lower: Lower triangle of A is stored.
///
/// @param[in] n
///     The number of linear equations, i.e., the order of the
///     matrix A. n >= 0.
///
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrix B. nrhs >= 0.
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array.
///     On entry, the symmetric matrix A.
///     - If uplo = Upper, the leading
///     n-by-n upper triangular part of A contains the upper
///     triangular part of the matrix A, and the strictly lower
///     triangular part of A is not referenced.
///
///     - If uplo = Lower, the
///     leading n-by-n lower triangular part of A contains the lower
///     triangular part of the matrix A, and the strictly upper
///     triangular part of A is not referenced.
///
///     - On successful exit, the block diagonal matrix D and the
///     multipliers used to obtain the factor U or L from the
///     factorization \f$ A = U D U^T \f$ or \f$ A = L D L^T \f$ as computed by
///     `lapack::sytrf`.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,n).
///
/// @param[out] ipiv
///     The vector ipiv of length n.
///     Details of the interchanges and the block structure of D, as
///     determined by `lapack::sytrf`. If ipiv(k) > 0, then rows and columns
///     k and ipiv(k) were interchanged, and D(k,k) is a 1-by-1
///     diagonal block.
///     - If uplo = Upper and ipiv(k) = ipiv(k-1) < 0,
///     then rows and columns k-1 and -ipiv(k) were interchanged and
///     D(k-1:k,k-1:k) is a 2-by-2 diagonal block.
///
///     - If uplo = Lower and
///     ipiv(k) = ipiv(k+1) < 0, then rows and columns k+1 and
///     -ipiv(k) were interchanged and D(k:k+1,k:k+1) is a 2-by-2
///     diagonal block.
///
/// @param[in,out] B
///     The n-by-nrhs matrix B, stored in an ldb-by-nrhs array.
///     On entry, the n-by-nrhs right hand side matrix B.
///     On successful exit, the n-by-nrhs solution matrix X.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,n).
///
/// @retval = 0: successful exit
/// @retval > 0: if return value = i, D(i,i) is exactly zero. The factorization
///              has been completed, but the block diagonal matrix D is
///              exactly singular, so the solution could not be computed.
///
/// @ingroup sysv
int64_t sysv(
    lapack::Uplo uplo, int64_t n, int64_t nrhs,
    std::complex<double>* A, int64_t lda,
    int64_t* ipiv,
    std::complex<double>* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char uplo_ = uplo2char( uplo );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( (n) );
        blas_int* ipiv_ptr = &ipiv_[0];
    #else
        blas_int* ipiv_ptr = ipiv;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_zsysv( &uplo_, &n_, &nrhs_, A, &lda_, ipiv_ptr, B, &ldb_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );

    LAPACK_zsysv( &uplo_, &n_, &nrhs_, A, &lda_, ipiv_ptr, B, &ldb_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    #if 1
        std::copy( ipiv_.begin(), ipiv_.end(), ipiv );
    #endif
    return info_;
}

}  // namespace lapack
