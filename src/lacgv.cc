#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup auxiliary
void lacgv(
    int64_t n,
    std::complex<float>* x, int64_t incx )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int incx_ = (blas_int) incx;

    LAPACK_clacgv( &n_, x, &incx_ );
}

// -----------------------------------------------------------------------------
/// Conjugates a complex vector of length n.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
/// Real precisions are dummy inline functions that do nothing,
/// to facilitate templating.
///
/// @param[in] n
///     The length of the vector x. n >= 0.
///
/// @param[in,out] x
///     The vector x of length n, stored in an array of length 1+(n-1)*abs(incx).
///     On entry, the vector of length n to be conjugated.
///     On exit, x is overwritten with conj(x).
///
/// @param[in] incx
///     The spacing between successive elements of x.
///
/// @ingroup auxiliary
void lacgv(
    int64_t n,
    std::complex<double>* x, int64_t incx )
{
    // check for overflow
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( std::abs(n) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }
    blas_int n_ = (blas_int) n;
    blas_int incx_ = (blas_int) incx;

    LAPACK_zlacgv( &n_, x, &incx_ );
}

}  // namespace lapack
