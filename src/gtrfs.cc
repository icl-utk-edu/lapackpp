#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gtsv_computational
int64_t gtrfs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    float const* DL,
    float const* D,
    float const* DU,
    float const* DLF,
    float const* DF,
    float const* DUF,
    float const* DU2,
    int64_t const* ipiv,
    float const* B, int64_t ldb,
    float* X, int64_t ldx,
    float* ferr,
    float* berr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< float > work( (3*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_sgtrfs( &trans_, &n_, &nrhs_, DL, D, DU, DLF, DF, DUF, DU2, ipiv_ptr, B, &ldb_, X, &ldx_, ferr, berr, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gtsv_computational
int64_t gtrfs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    double const* DL,
    double const* D,
    double const* DU,
    double const* DLF,
    double const* DF,
    double const* DUF,
    double const* DU2,
    int64_t const* ipiv,
    double const* B, int64_t ldb,
    double* X, int64_t ldx,
    double* ferr,
    double* berr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< double > work( (3*n) );
    std::vector< blas_int > iwork( (n) );

    LAPACK_dgtrfs( &trans_, &n_, &nrhs_, DL, D, DU, DLF, DF, DUF, DU2, ipiv_ptr, B, &ldb_, X, &ldx_, ferr, berr, &work[0], &iwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gtsv_computational
int64_t gtrfs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    std::complex<float> const* DL,
    std::complex<float> const* D,
    std::complex<float> const* DU,
    std::complex<float> const* DLF,
    std::complex<float> const* DF,
    std::complex<float> const* DUF,
    std::complex<float> const* DU2,
    int64_t const* ipiv,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float>* X, int64_t ldx,
    float* ferr,
    float* berr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<float> > work( (2*n) );
    std::vector< float > rwork( (n) );

    LAPACK_cgtrfs( &trans_, &n_, &nrhs_, DL, D, DU, DLF, DF, DUF, DU2, ipiv_ptr, B, &ldb_, X, &ldx_, ferr, berr, &work[0], &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Improves the computed solution to a system of linear
/// equations when the coefficient matrix is tridiagonal, and provides
/// error bounds and backward error estimates for the solution.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] trans
///     Specifies the form of the system of equations:
///     - lapack::Op::NoTrans:   \f$ A   X = B \f$ (No transpose)
///     - lapack::Op::Trans:     \f$ A^T X = B \f$ (Transpose)
///     - lapack::Op::ConjTrans: \f$ A^H X = B \f$ (Conjugate transpose)
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrix B. nrhs >= 0.
///
/// @param[in] DL
///     The vector DL of length n-1.
///     The (n-1) subdiagonal elements of A.
///
/// @param[in] D
///     The vector D of length n.
///     The diagonal elements of A.
///
/// @param[in] DU
///     The vector DU of length n-1.
///     The (n-1) superdiagonal elements of A.
///
/// @param[in] DLF
///     The vector DLF of length n-1.
///     The (n-1) multipliers that define the matrix L from the
///     LU factorization of A as computed by `lapack::gttrf`.
///
/// @param[in] DF
///     The vector DF of length n.
///     The n diagonal elements of the upper triangular matrix U from
///     the LU factorization of A.
///
/// @param[in] DUF
///     The vector DUF of length n-1.
///     The (n-1) elements of the first superdiagonal of U.
///
/// @param[in] DU2
///     The vector DU2 of length n-2.
///     The (n-2) elements of the second superdiagonal of U.
///
/// @param[in] ipiv
///     The vector ipiv of length n.
///     The pivot indices; for 1 <= i <= n, row i of the matrix was
///     interchanged with row ipiv(i). ipiv(i) will always be either
///     i or i+1; ipiv(i) = i indicates a row interchange was not
///     required.
///
/// @param[in] B
///     The n-by-nrhs matrix B, stored in an ldb-by-nrhs array.
///     The right hand side matrix B.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,n).
///
/// @param[in,out] X
///     The n-by-nrhs matrix X, stored in an ldx-by-nrhs array.
///     On entry, the solution matrix X, as computed by `lapack::gttrs`.
///     On exit, the improved solution matrix X.
///
/// @param[in] ldx
///     The leading dimension of the array X. ldx >= max(1,n).
///
/// @param[out] ferr
///     The vector ferr of length nrhs.
///     The estimated forward error bound for each solution vector
///     X(j) (the j-th column of the solution matrix X).
///     If XTRUE is the true solution corresponding to X(j), ferr(j)
///     is an estimated upper bound for the magnitude of the largest
///     element in (X(j) - XTRUE) divided by the magnitude of the
///     largest element in X(j). The estimate is as reliable as
///     the estimate for RCOND, and is almost always a slight
///     overestimate of the true error.
///
/// @param[out] berr
///     The vector berr of length nrhs.
///     The componentwise relative backward error of each solution
///     vector X(j) (i.e., the smallest relative change in
///     any element of A or B that makes X(j) an exact solution).
///
/// @retval = 0: successful exit
///
/// @ingroup gtsv_computational
int64_t gtrfs(
    lapack::Op trans, int64_t n, int64_t nrhs,
    std::complex<double> const* DL,
    std::complex<double> const* D,
    std::complex<double> const* DU,
    std::complex<double> const* DLF,
    std::complex<double> const* DF,
    std::complex<double> const* DUF,
    std::complex<double> const* DU2,
    int64_t const* ipiv,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double>* X, int64_t ldx,
    double* ferr,
    double* berr )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldx) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    #if 1
        // 32-bit copy
        std::vector< blas_int > ipiv_( &ipiv[0], &ipiv[(n)] );
        blas_int const* ipiv_ptr = &ipiv_[0];
    #else
        blas_int const* ipiv_ptr = ipiv_;
    #endif
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldx_ = (blas_int) ldx;
    blas_int info_ = 0;

    // allocate workspace
    std::vector< std::complex<double> > work( (2*n) );
    std::vector< double > rwork( (n) );

    LAPACK_zgtrfs( &trans_, &n_, &nrhs_, DL, D, DU, DLF, DF, DUF, DU2, ipiv_ptr, B, &ldb_, X, &ldx_, ferr, berr, &work[0], &rwork[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
