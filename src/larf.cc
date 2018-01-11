#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup unitary_computational
void larf(
    lapack::Side side, int64_t m, int64_t n,
    float const* v, int64_t incv, float tau,
    float* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(incv) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int incv_ = (blas_int) incv;
    blas_int ldc_ = (blas_int) ldc;

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    std::vector< float > work( lwork );

    LAPACK_slarf( &side_, &m_, &n_, v, &incv_, &tau, C, &ldc_, &work[0] );
}

// -----------------------------------------------------------------------------
/// @ingroup unitary_computational
void larf(
    lapack::Side side, int64_t m, int64_t n,
    double const* v, int64_t incv, double tau,
    double* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(incv) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int incv_ = (blas_int) incv;
    blas_int ldc_ = (blas_int) ldc;

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    std::vector< double > work( lwork );

    LAPACK_dlarf( &side_, &m_, &n_, v, &incv_, &tau, C, &ldc_, &work[0] );
}

// -----------------------------------------------------------------------------
/// @ingroup unitary_computational
void larf(
    lapack::Side side, int64_t m, int64_t n,
    std::complex<float> const* v, int64_t incv, std::complex<float> tau,
    std::complex<float>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(incv) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int incv_ = (blas_int) incv;
    blas_int ldc_ = (blas_int) ldc;

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    std::vector< std::complex<float> > work( lwork );

    LAPACK_clarf( &side_, &m_, &n_, v, &incv_, &tau, C, &ldc_, &work[0] );
}

// -----------------------------------------------------------------------------
/// Applies a elementary reflector H to a m-by-n
/// matrix C, from either the left or the right. H is represented in the
/// form
///
///     \f[ H = I - \tau v v^H, \f]
///
/// where \f$ \tau \f$ is a scalar and v is a vector.
///
/// If \f$ \tau = 0, \f$ then H is taken to be the unit matrix.
///
/// To apply \f$ H^H, \f$ supply \f$ \text{conj}(\tau) \f$ instead of \f$ \tau. \f$
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] side
///     - lapack::Side::Left:  form \f$ H C \f$
///     - lapack::Side::Right: form \f$ C H \f$
///
/// @param[in] m
///     The number of rows of the matrix C.
///
/// @param[in] n
///     The number of columns of the matrix C.
///
/// @param[in] v
///     The vector v in the representation of H. v is not used if tau = 0.
///     - If side = Left,  the vector v of length 1 + (m-1)*abs(incv);
///     - if side = Right, the vector v of length 1 + (n-1)*abs(incv).
///
/// @param[in] incv
///     The increment between elements of v. incv != 0.
///
/// @param[in] tau
///     The value tau in the representation of H.
///
/// @param[in,out] C
///     The m-by-n matrix C, stored in an ldc-by-n array.
///     On entry, the m-by-n matrix C.
///     On exit, C is overwritten by the matrix \f$ H C \f$ if side = Left,
///     or \f$ C H \f$ if side = Right.
///
/// @param[in] ldc
///     The leading dimension of the array C. ldc >= max(1,m).
///
/// @ingroup unitary_computational
void larf(
    lapack::Side side, int64_t m, int64_t n,
    std::complex<double> const* v, int64_t incv, std::complex<double> tau,
    std::complex<double>* C, int64_t ldc )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        lapack_error_if( std::abs(m) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(n) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(incv) > std::numeric_limits<blas_int>::max() );
        lapack_error_if( std::abs(ldc) > std::numeric_limits<blas_int>::max() );
    }
    char side_ = side2char( side );
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int incv_ = (blas_int) incv;
    blas_int ldc_ = (blas_int) ldc;

    // from docs
    int64_t lwork = (side == Side::Left ? n : m);

    // allocate workspace
    std::vector< std::complex<double> > work( lwork );

    LAPACK_zlarf( &side_, &m_, &n_, v, &incv_, &tau, C, &ldc_, &work[0] );
}

}  // namespace lapack
