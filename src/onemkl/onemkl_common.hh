// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_ONEMKL_COMMON_H
#define LAPACK_ONEMKL_COMMON_H

#include "lapack/device.hh"

#include <sycl/detail/cl.h>  // For CL version
#include <sycl/sycl.hpp>

#define MKL_Complex8  lapack_complex_float
#define MKL_Complex16 lapack_complex_double

#include <oneapi/mkl.hpp>

namespace lapack {

inline const oneapi::mkl::job job2onemkl( lapack::Job job )
{
    if (job == lapack::Job::NoVec) return oneapi::mkl::job::novec;
    if (job == lapack::Job::Vec) return oneapi::mkl::job::vec;
    if (job == lapack::Job::UpdateVec) return oneapi::mkl::job::updatevec;
    if (job == lapack::Job::AllVec) return oneapi::mkl::job::allvec;
    if (job == lapack::Job::SomeVec) return oneapi::mkl::job::somevec;
    if (job == lapack::Job::OverwriteVec) return oneapi::mkl::job::overwritevec;
    return oneapi::mkl::job::novec;
}

} // namespace lapack

#endif // LAPACK_ONEMKL_COMMON_H
