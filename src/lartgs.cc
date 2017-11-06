#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
void lartgs(
    float x, float y, float sigma,
    float* cs,
    float* sn )
{

    LAPACK_slartgs( &x, &y, &sigma, cs, sn );
}

// -----------------------------------------------------------------------------
void lartgs(
    double x, double y, double sigma,
    double* cs,
    double* sn )
{

    LAPACK_dlartgs( &x, &y, &sigma, cs, sn );
}

}  // namespace lapack
