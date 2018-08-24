#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup heev_computational
int64_t upmtr(
    lapack::Side side, lapack::Uplo uplo, lapack::Op trans, int64_t m, int64_t n,
    std::complex<float> const* AP,
    std::complex<float> const* tau,
    std::complex<float>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char side_ = side2char( side );
    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int ldc_ = (lapack_int) ldc;
    lapack_int info_ = 0;

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    std::vector< std::complex<float> > work( max(1,lwork) );

    LAPACK_cupmtr(
        &side_, &uplo_, &trans_, &m_, &n_,
        (lapack_complex_float*) AP,
        (lapack_complex_float*) tau,
        (lapack_complex_float*) C, &ldc_,
        (lapack_complex_float*) &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

// -----------------------------------------------------------------------------
/// Multiplies the general m-by-n matrix C by Q from `lapack::hptrd` as follows:
///
/// - side = left,  trans = NoTrans:   \f$ Q C \f$
/// - side = right, trans = NoTrans:   \f$ C Q \f$
/// - side = left,  trans = ConjTrans: \f$ Q^H C \f$
/// - side = right, trans = ConjTrans: \f$ C Q^H \f$
///
/// where Q is a unitary matrix of order nq, with nq = m if
/// side = Left and nq = n if side = Right. Q is defined as the product of
/// nq-1 elementary reflectors, as returned by `lapack::hptrd` using packed
/// storage:
///
/// - if uplo = Upper, \f$ Q = H(nq-1) \dots H(2) H(1); \f$
/// - if uplo = Lower, \f$ Q = H(1) H(2) \dots H(nq-1). \f$
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// For real matrices, this is an alias for `lapack::opmtr`.
///
/// @param[in] side
///     - lapack::Side::Left:  apply \f$ Q \f$ or \f$ Q^H \f$ from the Left;
///     - lapack::Side::Right: apply \f$ Q \f$ or \f$ Q^H \f$ from the Right.
///
/// @param[in] uplo
///     - lapack::Uplo::Upper: Upper triangular packed storage used in previous
///         call to `lapack::hptrd`;
///     - lapack::Uplo::Lower: Lower triangular packed storage used in previous
///         call to `lapack::hptrd`.
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
/// @param[in] AP
///     The vectors which define the elementary reflectors, as
///     returned by `lapack::hptrd`. AP is modified by the routine but
///     restored on exit.
///     - If side = Left,  AP is of length m*(m+1)/2;
///     - if side = Right, AP is of length n*(n+1)/2.
///
/// @param[in] tau
///     tau(i) must contain the scalar factor of the elementary
///     reflector H(i), as returned by `lapack::hptrd`.
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
int64_t upmtr(
    lapack::Side side, lapack::Uplo uplo, lapack::Op trans, int64_t m, int64_t n,
    std::complex<double> const* AP,
    std::complex<double> const* tau,
    std::complex<double>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<lapack_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<lapack_int>::max() );
    }
    char side_ = side2char( side );
    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int ldc_ = (lapack_int) ldc;
    lapack_int info_ = 0;

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    std::vector< std::complex<double> > work( max(1,lwork) );

    LAPACK_zupmtr(
        &side_, &uplo_, &trans_, &m_, &n_,
        (lapack_complex_double*) AP,
        (lapack_complex_double*) tau,
        (lapack_complex_double*) C, &ldc_,
        (lapack_complex_double*) &work[0], &info_ );
    if (info_ < 0) {
        throw Error();
    }
    return info_;
}

}  // namespace lapack
