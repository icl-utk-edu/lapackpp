#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup heev_computational
int64_t unmtr(
    lapack::Side side, lapack::Uplo uplo, lapack::Op trans, int64_t m, int64_t n,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* tau,
    std::complex<float>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // for complex, map Trans to ConjTrans
    if (trans_ == 'T')
        trans_ = 'C';

    // query for workspace size
    std::complex<float> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_cunmtr( &side_, &uplo_, &trans_, &m_, &n_, A, &lda_, tau, C, &ldc_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork_ );

    LAPACK_cunmtr( &side_, &uplo_, &trans_, &m_, &n_, A, &lda_, tau, C, &ldc_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Multiplies the general m-by-n matrix C by Q from `lapack::hetrd` as follows:
///
/// - side = left,  trans = NoTrans:   \f$ Q C \f$
/// - side = right, trans = NoTrans:   \f$ C Q \f$
/// - side = left,  trans = ConjTrans: \f$ Q^H C \f$
/// - side = right, trans = ConjTrans: \f$ C Q^H \f$
///
/// where Q is a unitary matrix of order nq, with nq = m if
/// side = Left and nq = n if side = Right. Q is defined as the product of
/// nq-1 elementary reflectors, as returned by `lapack::hetrd`:
///
/// - if uplo = Upper, \f$ Q = H(nq-1) \dots H(2) H(1); \f$
/// - if uplo = Lower, \f$ Q = H(1) H(2) \dots H(nq-1). \f$
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this is an alias for `lapack::ormtr`.
///
/// @param[in] side
///     - lapack::Side::Left:  apply \f$ Q \f$ or \f$ Q^H \f$ from the Left;
///     - lapack::Side::Right: apply \f$ Q \f$ or \f$ Q^H \f$ from the Right.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangle of A contains elementary reflectors
///         from `lapack::hetrd`;
///     - lapack::Uplo::Lower: Lower triangle of A contains elementary reflectors
///         from `lapack::hetrd`.
///
/// @param[in] trans
///     - lapack::Op::NoTrans:   No transpose, apply \f$ Q \f$;
///     - lapack::Op::ConjTrans: Conjugate transpose, apply \f$ Q^H \f$.
///
/// @param[in] m
///     The number of rows of the matrix C. m >= 0.
///
/// @param[in] n
///     The number of columns of the matrix C. n >= 0.
///
/// @param[in] A
///     The vectors which define the elementary reflectors, as
///     returned by `lapack::hetrd`.
///     - If side = Left,  the m-by-m matrix A, stored in an lda-by-m array;
///     - if side = Right, the n-by-n matrix A, stored in an lda-by-n array.
///
/// @param[in] lda
///     The leading dimension of the array A.
///     - If side = Left,  lda >= max(1,m);
///     - If side = Right, lda >= max(1,n).
///
/// @param[in] tau
///     tau(i) must contain the scalar factor of the elementary
///     reflector H(i), as returned by `lapack::hetrd`.
///     - If side = Left,  the vector tau of length m-1;
///     - if side = Right, the vector tau of length n-1.
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
/// @ingroup heev_computational
int64_t unmtr(
    lapack::Side side, lapack::Uplo uplo, lapack::Op trans, int64_t m, int64_t n,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* tau,
    std::complex<double>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(m) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(lda) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldc_ = (blas_int) ldc;
    blas_int info_ = 0;

    // for complex, map Trans to ConjTrans
    if (trans_ == 'T')
        trans_ = 'C';

    // query for workspace size
    std::complex<double> qry_work[1];
    blas_int ineg_one = -1;
    LAPACK_zunmtr( &side_, &uplo_, &trans_, &m_, &n_, A, &lda_, tau, C, &ldc_, qry_work, &ineg_one, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    blas_int lwork_ = real(qry_work[0]);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork_ );

    LAPACK_zunmtr( &side_, &uplo_, &trans_, &m_, &n_, A, &lda_, tau, C, &ldc_, &work[0], &lwork_, &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
