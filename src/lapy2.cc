#include "lapack.hh"
#include "lapack_fortran.h"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
float lapy2(
    float x, float y )
{

    return LAPACK_slapy2( &x, &y );
}

// -----------------------------------------------------------------------------
double lapy2(
    double x, double y )
{

    return LAPACK_dlapy2( &x, &y );
}

}  // namespace lapack
