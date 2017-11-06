#include "lapack_fortran.h"
#include "lapack_util.hh"

#include <vector>

namespace lapack {

using std::max;
using std::min;
using blas::real;

// -----------------------------------------------------------------------------
void lartgp(
    float f, float g,
    float* cs,
    float* sn,
    float* r )
{

    LAPACK_slartgp( &f, &g, cs, sn, r );
}

// -----------------------------------------------------------------------------
void lartgp(
    double f, double g,
    double* cs,
    double* sn,
    double* r )
{

    LAPACK_dlartgp( &f, &g, cs, sn, r );
}

}  // namespace lapack
