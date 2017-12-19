#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup ggls
int64_t gglse(
    int64_t m, int64_t n, int64_t p,
    float* A, int64_t lda,
    float* B, int64_t ldb,
    float* C,
    float* D,
    float* X )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(p) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int p_ = (blas_int) p;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_sgglse( &m_, &n_, &p_, A, &lda_, B, &ldb_, C, D, X, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_sgglse( &m_, &n_, &p_, A, &lda_, B, &ldb_, C, D, X, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ggls
int64_t gglse(
    int64_t m, int64_t n, int64_t p,
    double* A, int64_t lda,
    double* B, int64_t ldb,
    double* C,
    double* D,
    double* X )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(p) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int p_ = (blas_int) p;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dgglse( &m_, &n_, &p_, A, &lda_, B, &ldb_, C, D, X, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dgglse( &m_, &n_, &p_, A, &lda_, B, &ldb_, C, D, X, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup ggls
int64_t gglse(
    int64_t m, int64_t n, int64_t p,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb,
    std::complex<float>* C,
    std::complex<float>* D,
    std::complex<float>* X )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(p) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int p_ = (blas_int) p;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_cgglse( &m_, &n_, &p_, A, &lda_, B, &ldb_, C, D, X, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );

    LAPACK_cgglse( &m_, &n_, &p_, A, &lda_, B, &ldb_, C, D, X, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Solves the linear equality-constrained least squares (LSE)
/// problem:
///
///     \f[ \min_x || c - A x ||_2 \text{ subject to } B x = d \f]
///
/// where A is an m-by-n matrix, B is a p-by-n matrix, c is a given
/// m-vector, and d is a given p-vector. It is assumed that
/// p <= n <= m+p, and
///
/// rank(B) = p and
/// rank\f$\left( \left[ \begin{array}{c} A \\ B \end{array} \right] \right) = n. \f$
///
/// These conditions ensure that the LSE problem has a unique solution,
/// which is obtained using a generalized RQ factorization of the
/// matrices (B, A) given by
///
///     \f[ B = \left[ 0 \;\; R \right] Q, \quad A = Z T Q. \f]
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] m
///     The number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrices A and B. n >= 0.
///
/// @param[in] p
///     The number of rows of the matrix B. 0 <= p <= n <= m+p.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     On entry, the m-by-n matrix A.
///     On exit, the elements on and above the diagonal of the array
///     contain the min(m,n)-by-n upper trapezoidal matrix T.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @param[in,out] B
///     The p-by-n matrix B, stored in an ldb-by-n array.
///     On entry, the p-by-n matrix B.
///     On exit, the upper triangle of the subarray B(1:p,n-p+1:n)
///     contains the p-by-p upper triangular matrix R.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= max(1,p).
///
/// @param[in,out] C
///     The vector C of length m.
///     On entry, C contains the right hand side vector for the
///     least squares part of the LSE problem.
///     On exit, the residual sum of squares for the solution
///     is given by the sum of squares of elements n-p+1 to m of
///     vector C.
///
/// @param[in,out] D
///     The vector D of length p.
///     On entry, D contains the right hand side vector for the
///     constrained equation.
///     On exit, D is destroyed.
///
/// @param[out] X
///     The vector X of length n.
///     On exit, X is the solution of the LSE problem.
///
/// @retval = 0: successful exit.
/// @retval = 1: the upper triangular factor R associated with B in the
///     generalized RQ factorization of the pair (B, A) is
///     singular, so that rank(B) < p; the least squares
///     solution could not be computed.
/// @retval = 2: the (n-p) by (n-p) part of the upper trapezoidal factor
///     T associated with A in the generalized RQ factorization
///     of the pair (B, A) is singular, so that
///     rank\f$\left( \left[ \begin{array}{c} A \\ B \end{array} \right] \right) < n \f$;
///     the least squares solution could not be computed.
///
/// @ingroup ggls
int64_t gglse(
    int64_t m, int64_t n, int64_t p,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb,
    std::complex<double>* C,
    std::complex<double>* D,
    std::complex<double>* X )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(p) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int p_ = (blas_int) p;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_zgglse( &m_, &n_, &p_, A, &lda_, B, &ldb_, C, D, X, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );

    LAPACK_zgglse( &m_, &n_, &p_, A, &lda_, B, &ldb_, C, D, X, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
