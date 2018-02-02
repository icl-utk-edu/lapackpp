#include "matrix_opts.hh"

using libtest::ParamType;

matrix_opts::matrix_opts():
	name ("test-matrix-name",  0, ParamType::List, "rand", "", "test matrix type" ),
	cond  ("test-matrix-cond",  16, 4, ParamType::Value, 0.0, 1.0, std::numeric_limits<double>::infinity(), "matrix A condition number" ),
	condD ("test-matrix-condD", 16, 4, ParamType::Value, 1.0, 1.0, std::numeric_limits<double>::infinity(), "matrix D condition number" )
{
    this->iseed[0]  = 0;
    this->iseed[1]  = 1;
    this->iseed[2]  = 2;
    this->iseed[3]  = 3;
}
