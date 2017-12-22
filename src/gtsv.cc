#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gtsv
int64_t gtsv(
    int64_t n, int64_t nrhs,
    float* DL,
    float* D,
    float* DU,
    float* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    LAPACK_sgtsv( &n_, &nrhs_, DL, D, DU, B, &ldb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gtsv
int64_t gtsv(
    int64_t n, int64_t nrhs,
    double* DL,
    double* D,
    double* DU,
    double* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    LAPACK_dgtsv( &n_, &nrhs_, DL, D, DU, B, &ldb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gtsv
int64_t gtsv(
    int64_t n, int64_t nrhs,
    std::complex<float>* DL,
    std::complex<float>* D,
    std::complex<float>* DU,
    std::complex<float>* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    LAPACK_cgtsv( &n_, &nrhs_, DL, D, DU, B, &ldb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Solves the equation
///
///     \f$ A X = B, \f$
///
/// where A is an n-by-n tridiagonal matrix, by Gaussian elimination with
/// partial pivoting.
///
/// Note that the equation \f$ A^T X = B \f$ may be solved by interchanging the
/// order of the arguments DU and DL.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] n
///     The order of the matrix A. n >= 0.
///
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of columns
///     of the matrix B. nrhs >= 0.
///
/// @param[in,out] DL
///     The vector DL of length n-1.
///     On entry, DL must contain the (n-1) subdiagonal elements of A.
///     On exit, DL is overwritten by the (n-2) elements of the
///     second superdiagonal of the upper triangular matrix U from
///     the LU factorization of A, in DL(1), ..., DL(n-2).
///
/// @param[in,out] D
///     The vector D of length n.
///     On entry, D must contain the diagonal elements of A.
///     On exit, D is overwritten by the n diagonal elements of U.
///
/// @param[in,out] DU
///     The vector DU of length n-1.
///     On entry, DU must contain the (n-1) superdiagonal elements of A.
///     On exit, DU is overwritten by the (n-1) elements of the first
///     superdiagonal of U.
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
/// @retval > 0: if return value = i, U(i,i) is exactly zero, and the solution
///     has not been computed. The factorization has not been
///     completed unless i = n.
///
/// @ingroup gtsv
int64_t gtsv(
    int64_t n, int64_t nrhs,
    std::complex<double>* DL,
    std::complex<double>* D,
    std::complex<double>* DU,
    std::complex<double>* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    LAPACK_zgtsv( &n_, &nrhs_, DL, D, DU, B, &ldb_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
