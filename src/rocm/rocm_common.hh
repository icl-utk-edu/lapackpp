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

#endif // LAPACK_ROCM_COMMON_H
