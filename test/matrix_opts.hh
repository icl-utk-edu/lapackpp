#ifndef MATRIX_OPTS_HH
#define MATRIX_OPTS_HH

#include "libtest.hh"

#define MAX_NTEST 1050
#define MAXGPUS 8

class matrix_opts
{
public:
    // constructor
    matrix_opts();

    int64_t verbose;

    // LAPACK test matrix generation
    libtest::ParamString name;
    libtest::ParamDouble cond;
    libtest::ParamDouble condD;
    int64_t     iseed[4];
};

#endif  // #ifndef MATRIX_OPTS_HH
