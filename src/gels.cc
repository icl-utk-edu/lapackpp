#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gels
int64_t gels(
    lapack::Op trans, int64_t m, int64_t n, int64_t nrhs,
    float* A, int64_t lda,
    float* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // for real, map ConjTrans to Trans
    if (trans_ == 'C')
        trans_ = 'T';

    // query for workspace size
    float qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_sgels( &trans_, &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< float > work( lwork_ );

    LAPACK_sgels( &trans_, &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gels
int64_t gels(
    lapack::Op trans, int64_t m, int64_t n, int64_t nrhs,
    double* A, int64_t lda,
    double* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // for real, map ConjTrans to Trans
    if (trans_ == 'C')
        trans_ = 'T';

    // query for workspace size
    double qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_dgels( &trans_, &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< double > work( lwork_ );

    LAPACK_dgels( &trans_, &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// @ingroup gels
int64_t gels(
    lapack::Op trans, int64_t m, int64_t n, int64_t nrhs,
    std::complex<float>* A, int64_t lda,
    std::complex<float>* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_cgels( &trans_, &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );

    LAPACK_cgels( &trans_, &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Solves overdetermined or underdetermined complex linear systems
/// involving an m-by-n matrix A, or its conjugate-transpose, using a QR
/// or LQ factorization of A. It is assumed that A has full rank.
///
/// The following options are provided:
///
/// 1. If trans = NoTrans and m >= n: find the least squares solution of
///     an overdetermined system, i.e., solve the least squares problem
///     minimize \f$ || B - A X ||_2 \f$.
///
/// 2. If trans = NoTrans and m < n: find the minimum norm solution of
///     an underdetermined system \f$ A X = B \f$.
///
/// 3. If trans = ConjTrans and m >= n: find the minimum norm solution of
///     an underdetermined system \f$ A^H X = B \f$.
///
/// 4. If trans = ConjTrans and m < n: find the least squares solution of
///     an overdetermined system, i.e., solve the least squares problem
///     minimize \f$ || B - A^H X ||_2 \f$.
///
/// Several right hand side vectors b and solution vectors x can be
/// handled in a single call; they are stored as the columns of the
/// m-by-nrhs right hand side matrix B and the n-by-nrhs solution
/// matrix X.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] trans
///     - lapack::Op::NoTrans:   the linear system involves \f$ A   \f$;
///     - lapack::Op::ConjTrans: the linear system involves \f$ A^H \f$.
///     - lapack::Op::Trans:     the linear system involves \f$ A^T \f$.
///     \n
///     For real matrices, Trans = ConjTrans.
///     For complex matrices, Trans is illegal.
///
/// @param[in] m
///     The number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix A. n >= 0.
///
/// @param[in] nrhs
///     The number of right hand sides, i.e., the number of
///     columns of the matrices B and X. nrhs >= 0.
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array.
///     On entry, the m-by-n matrix A.
///     - If m >= n, A is overwritten by details of its QR
///     factorization as returned by `lapack::geqrf`;
///
///     - If m < n, A is overwritten by details of its LQ
///     factorization as returned by `lapack::gelqf`.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,m).
///
/// @param[in,out] B
///     The max(m,n)-by-nrhs matrix B, stored in an ldb-by-nrhs array.
///     On entry, the matrix B of right hand side vectors, stored
///     columnwise; B is m-by-nrhs if trans = NoTrans, or n-by-nrhs
///     if trans = ConjTrans.
///     On successful exit, B is overwritten by the solution
///     vectors, stored columnwise:
///     - If trans = NoTrans and m >= n, rows 1 to n of B contain the least
///     squares solution vectors; the residual sum of squares for the
///     solution in each column is given by the sum of squares of the
///     modulus of elements n+1 to m in that column;
///
///     - If trans = NoTrans and m < n, rows 1 to n of B contain the
///     minimum norm solution vectors;
///
///     - If trans = ConjTrans and m >= n, rows 1 to m of B contain the
///     minimum norm solution vectors;
///
///     - If trans = ConjTrans and m < n, rows 1 to m of B contain the
///     least squares solution vectors; the residual sum of squares
///     for the solution in each column is given by the sum of
///     squares of the modulus of elements m+1 to n in that column.
///
/// @param[in] ldb
///     The leading dimension of the array B. ldb >= MAX(1,m,n).
///
/// @retval = 0: successful exit
/// @retval > 0: if return value = i, the i-th diagonal element of the
///     triangular factor of A is zero, so that A does not have
///     full rank; the least squares solution could not be
///     computed.
///
/// @ingroup gels
int64_t gels(
    lapack::Op trans, int64_t m, int64_t n, int64_t nrhs,
    std::complex<double>* A, int64_t lda,
    std::complex<double>* B, int64_t ldb )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(nrhs) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldb) > std::numeric_limits<blas_int>::max() );
    }
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int nrhs_ = (blas_int) nrhs;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_zgels( &trans_, &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );

    LAPACK_zgels( &trans_, &m_, &n_, &nrhs_, A, &lda_, B, &ldb_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
