// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_INTERNAL_HH
#define LAPACK_INTERNAL_HH

#include "lapack/util.hh"

namespace lapack {

//------------------------------------------------------------------------------
/// @see to_lapack_int
///
inline lapack_int to_lapack_int_( int64_t x, const char* x_str )
{
    if constexpr (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if_msg( x > std::numeric_limits<lapack_int>::max(), "%s", x_str );
    }
    return lapack_int( x );
}

//----------------------------------------
/// Convert int64_t to lapack_int.
/// If lapack_int is 64-bit, this does nothing.
/// If lapack_int is 32-bit, throws if x > INT_MAX, so conversion would overflow.
///
/// Note this is in src/lapack_internal.hh, so this macro won't pollute
/// the namespace when apps #include <lapack.hh>.
///
#define to_lapack_int( x ) lapack::to_lapack_int_( x, #x )

}  // namespace lapack

#endif // LAPACK_INTERNAL_HH
