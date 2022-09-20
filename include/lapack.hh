// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_HH
#define LAPACK_HH

#include "lapack/defines.h"

// Version is updated by make_release.py; DO NOT EDIT.
// Version 2022.07.00
#define LAPACKPP_VERSION 20220700

namespace lapack {

int lapackpp_version();
const char* lapackpp_id();

}  // namespace lapack

#include "lapack/wrappers.hh"

#endif // LAPACK_HH
