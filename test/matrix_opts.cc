#include "matrix_opts.hh"

matrix_opts::matrix_opts (matrix_opts_t opts)
{
    this->matrix    = "rand";
    this->cond      = 0;  // zero means cond = sqrt( 1/eps ), which varies by precision
    this->condD     = 1;

    this->iseed[0]  = 1;
    this->iseed[1]  = 3;
    this->iseed[2]  = 5;
    this->iseed[3]  = 7;

}
