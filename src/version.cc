// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack.hh"
#include "lapack_internal.hh"

namespace lapack {

//------------------------------------------------------------------------------
/// @return LAPACK++ version.
/// Version is integer of form yyyymmrr, where yyyy is year, mm is month,
/// and rr is release counter within month, starting at 00.
///
int lapackpp_version()
{
    return LAPACKPP_VERSION;
}

// LAPACKPP_ID is the Mercurial or git commit hash ID, either
// defined by `git rev-parse --short HEAD` in Makefile,
// or defined here by make_release.py for release tar files. DO NOT EDIT.
#ifndef LAPACKPP_ID
#define LAPACKPP_ID "unknown"
#endif

//------------------------------------------------------------------------------
/// @return LAPACK++ Mercurial or git commit hash ID.
///
const char* lapackpp_id()
{
    return LAPACKPP_ID;
}

}  // namespace lapack
