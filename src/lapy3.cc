#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
/// @ingroup auxiliary
float lapy3(
    float x, float y, float z )
{
    return LAPACK_slapy3( &x, &y, &z );
}

// -----------------------------------------------------------------------------
/// Returns \f$ \sqrt{ x^2 + y^2 + z^2 }, \f$ taking care not to cause
/// unnecessary overflow.
///
/// Overloaded versions are available for
/// `float`, `double`, `std::complex<float>`, and `std::complex<double>`.
///
/// @param[in] x
///
/// @param[in] y
///
/// @param[in] z
///     x, y and z specify the values x, y and z.
///
/// @ingroup auxiliary
double lapy3(
    double x, double y, double z )
{
    return LAPACK_dlapy3( &x, &y, &z );
}

}  // namespace lapack
