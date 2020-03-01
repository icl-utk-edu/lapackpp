#ifndef LAPACK_HH
#define LAPACK_HH

// Version is updated by make_release.py; DO NOT EDIT.
// Version 0000.00.00
#define LAPACKPP_VERSION 00000000

namespace lapack {

int lapackpp_version();
const char* lapackpp_id();

}  // namespace lapack

#include "lapack/wrappers.hh"

#endif // LAPACK_HH
