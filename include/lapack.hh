// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_HH
#define LAPACK_HH

#include "lapack/defines.h"

// Version is updated by make_release.py; DO NOT EDIT.
// Version 2025.05.28
#define LAPACKPP_VERSION 20250528

namespace lapack {

int lapackpp_version();
const char* lapackpp_id();

}  // namespace lapack

#include "lapack/wrappers.hh"

#endif // LAPACK_HH
