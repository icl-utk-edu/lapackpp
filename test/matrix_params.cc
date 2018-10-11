#include "matrix_params.hh"

using libtest::ParamType;

const double inf = std::numeric_limits<double>::infinity();

// -----------------------------------------------------------------------------
/// Construct MatrixParams
MatrixParams::MatrixParams():
    verbose( 0 ),
    iseed {98, 108, 97, 115},

    //          name,    w, p, type,            default,             min, max, help
    kind      ("matrix", 0,    ParamType::List, "rand",                        "test matrix kind; see 'test --help-matrix'" ),
    cond      ("cond",   0, 1, ParamType::List, libtest::no_data_flag, 0, inf, "matrix condition number" ),
    cond_used ("cond",   0, 1, ParamType::List, libtest::no_data_flag, 0, inf, "actual condition number used" ),
    condD     ("condD",  0, 1, ParamType::List, libtest::no_data_flag, 0, inf, "matrix D condition number" )
{}

// -----------------------------------------------------------------------------
/// Marks matrix params as used.
void MatrixParams::mark()
{
    kind();
    cond();
    condD();
}
