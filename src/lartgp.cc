#include "lapack.hh"
#include "lapack/fortran.h"

#if LAPACK_VERSION >= 30300  // >= v3.3

#include <vector>

namespace lapack {

using blas::max;
using blas::min;
using blas::real;

// -----------------------------------------------------------------------------
void lartgp(
    float f, float g,
    float* cs,
    float* sn,
    float* r )
{

    LAPACK_slartgp(
        &f, &g, cs, sn, r );
}

// -----------------------------------------------------------------------------
void lartgp(
    double f, double g,
    double* cs,
    double* sn,
    double* r )
{

    LAPACK_dlartgp(
        &f, &g, cs, sn, r );
}

}  // namespace lapack

#endif  // LAPACK >= v3.3
