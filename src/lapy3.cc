#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
float lapy3(
    float x, float y, float z )
{

    return LAPACK_slapy3( &x, &y, &z );
}

// -----------------------------------------------------------------------------
double lapy3(
    double x, double y, double z )
{

    return LAPACK_dlapy3( &x, &y, &z );
}

}  // namespace lapack
