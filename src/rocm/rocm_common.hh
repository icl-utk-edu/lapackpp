// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_ROCM_COMMON_H
#define LAPACK_ROCM_COMMON_H

#include "lapack/device.hh"

#include <hip/hip_runtime.h>

// Headers moved in ROCm 5.2
#if HIP_VERSION >= 50200000
    #include <rocsolver/rocsolver.h>
#else
    #include <rocsolver.h>
#endif

namespace lapack {

// todo: Move to blaspp/src/onemkl_wrappers.cc as blas::internal::jobz2onemkl()?
// should we call it job2evect_rocsolver ??
inline const rocblas_evect  job2eigmode_rocsolver(lapack::Job jobz) {
    if (jobz == lapack::Job::NoVec) return rocblas_evect_none;
    if (jobz == lapack::Job::Vec) return rocblas_evect_original;
    return rocblas_evect_none;
}

} // namespace lapack

#endif // LAPACK_ROCM_COMMON_H
