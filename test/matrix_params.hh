#ifndef MATRIX_PARAMS_HH
#define MATRIX_PARAMS_HH

#include "libtest.hh"

// =============================================================================
class MatrixParams
{
public:
    MatrixParams();

    void mark();

    int64_t verbose;
    int64_t iseed[4];

    // ---- test matrix generation parameters
    libtest::ParamString kind;
    libtest::ParamScientific cond, cond_used;
    libtest::ParamScientific condD;
};

#endif  // #ifndef MATRIX_PARAMS_HH
