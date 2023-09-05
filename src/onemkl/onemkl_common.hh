// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_ONEMKL_COMMON_H
#define LAPACK_ONEMKL_COMMON_H

#include "lapack/device.hh"

#include <sycl/detail/cl.h>  // For CL version
#include <sycl.hpp>

#define MKL_Complex8  lapack_complex_float
#define MKL_Complex16 lapack_complex_double

#include <oneapi/mkl.hpp>

namespace lapack {

// todo: Move to blaspp/src/onemkl_wrappers.cc as blas::internal::jobz2onemkl()?
inline const oneapi::mkl::job jobz2onemkl(lapack::Job jobz) {
    if (jobz == lapack::Job::NoVec) return oneapi::mkl::job::novec;
    if (jobz == lapack::Job::Vec) return oneapi::mkl::job::vec;
    if (jobz == lapack::Job::UpdateVec) return oneapi::mkl::job::updatevec;
    if (jobz == lapack::Job::AllVec) return oneapi::mkl::job::allvec;
    if (jobz == lapack::Job::SomeVec) return oneapi::mkl::job::somevec;
    if (jobz == lapack::Job::OverwriteVec) return oneapi::mkl::job::overwritevec;
    return oneapi::mkl::job::novec;
}

} // namespace lapack

#endif // LAPACK_ONEMKL_COMMON_H
