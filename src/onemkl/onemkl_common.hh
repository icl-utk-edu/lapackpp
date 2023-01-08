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

#endif // LAPACK_ONEMKL_COMMON_H
