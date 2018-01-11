#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup gerqf
int64_t unmrq(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t k,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* tau,
    std::complex<float>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<float> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_cunmrq( &side_, &trans_, &m_, &n_, &k_, A, &lda_, tau, C, &ldc_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );

    LAPACK_cunmrq( &side_, &trans_, &m_, &n_, &k_, A, &lda_, tau, C, &ldc_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Multiplies the general m-by-n matrix C by Q from `lapack::gerqf` as follows:
///
/// - side = Left,  trans = NoTrans:   \f$ Q C \f$
/// - side = Right, trans = NoTrans:   \f$ C Q \f$
/// - side = Left,  trans = ConjTrans: \f$ Q^H C \f$
/// - side = Right, trans = ConjTrans: \f$ C Q^H \f$
///
/// where Q is a unitary matrix defined as the product of k
/// elementary reflectors, as returned by `lapack::gerqf`:
///
///     \f[ Q = H(1)^H H(2)^H \dots H(k)^H. \f]
///
/// Q is of order m if side = Left and of order n if side = Right.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this is an alias for `lapack::ormrq`.
///
/// @param[in] side
///     - lapack::Side::Left:  apply \f$ Q \f$ or \f$ Q^H \f$ from the Left;
///     - lapack::Side::Right: apply \f$ Q \f$ or \f$ Q^H \f$ from the Right.
///
/// @param[in] trans
///     - lapack::Op::NoTrans: No transpose, apply \f$ Q \f$;
///     - lapack::Op::ConjTrans: Transpose, apply \f$ Q^H \f$.
///
/// @param[in] m
///     The number of rows of the matrix C. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix C. n >= 0.
///
/// @param[in] k
///     The number of elementary reflectors whose product defines
///     the matrix Q.
///     - If side = Left,  m >= k >= 0;
///     - if side = Right, n >= k >= 0.
///
/// @param[in] A
///     - If side = Left,  the k-by-m matrix A, stored in an lda-by-m array;
///     - if side = Right, the k-by-n matrix A, stored in an lda-by-n array.
///     \n
///     The i-th row must contain the vector which defines the
///     elementary reflector H(i), for i = 1, 2, ..., k, as returned by
///     `lapack::gerqf` in the last k rows of its array argument A.
///
/// @param[in] lda
///     The leading dimension of the array A. lda >= max(1,k).
///
/// @param[in] tau
///     The vector tau of length k.
///     tau(i) must contain the scalar factor of the elementary
///     reflector H(i), as returned by `lapack::gerqf`.
///
/// @param[in,out] C
///     The m-by-n matrix C, stored in an ldc-by-n array.
///     On entry, the m-by-n matrix C.
///     On exit, C is overwritten by
///     \f$ Q C \f$ or \f$ Q^H C \f$ or \f$ C Q^H \f$ or \f$ C Q \f$.
///
/// @param[in] ldc
///     The leading dimension of the array C. ldc >= max(1,m).
///
/// @retval = 0: successful exit
///
/// @ingroup gerqf
int64_t unmrq(
    lapack::Side side, lapack::Op trans, int64_t m, int64_t n, int64_t k,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* tau,
    std::complex<double>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(k) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int k_ = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // query for workspace size
    std::complex<double> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_zunmrq( &side_, &trans_, &m_, &n_, &k_, A, &lda_, tau, C, &ldc_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );

    LAPACK_zunmrq( &side_, &trans_, &m_, &n_, &k_, A, &lda_, tau, C, &ldc_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
