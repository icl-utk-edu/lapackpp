// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_CONFIG_H
#define LAPACK_CONFIG_H

#include "blas/defines.h"
#include "lapack/defines.h"

#include <stdlib.h>
#include <stdint.h>

#if defined(BLAS_ILP64) && ! defined(LAPACK_ILP64)
    #define LAPACK_ILP64
#endif

#ifdef ACCELERATE_NEW_LAPACK
    // Apple's headers use std::complex in C++ and, for ILP64, `long`
    // (a distinct type from int64_t == long long on LP64 Darwin).
    #if defined(__cplusplus) && ! defined(LAPACK_COMPLEX_CPP)
        #define LAPACK_COMPLEX_CPP
    #endif
    #if defined(LAPACK_ILP64) && ! defined(ACCELERATE_LAPACK_ILP64)
        #define ACCELERATE_LAPACK_ILP64
    #endif
#endif

#ifndef lapack_int
    #if defined(LAPACK_ILP64) && defined(ACCELERATE_NEW_LAPACK)
        typedef long lapack_int;   // matches Apple's __LAPACK_int
    #elif defined(LAPACK_ILP64)
        typedef int64_t lapack_int;
    #else
        typedef int lapack_int;
    #endif
    // #define so we can check later with #ifdef.
    #define lapack_int lapack_int
#endif

#if defined(__cplusplus) && defined(LAPACK_ILP64)
    // lapack_int can be a distinct type from int64_t with the same size and
    // representation (long vs. long long with Apple's ILP64 Accelerate).
    // The wrappers cast integer-array pointers between the two, so the
    // sizes must agree.
    static_assert( sizeof(lapack_int) == sizeof(int64_t),
                   "lapack_int must be the same size as int64_t" );
#endif

#ifndef lapack_logical
    #define lapack_logical lapack_int
#endif

// f2c, hence MacOS Accelerate (before 13.3), returns double instead of float
// for sdot, slange, clange, etc.
// LAPACKE's lapack.h is missing #define to protect against multiple
// definitions, so use lapackpp prefix.
#if (defined(BLAS_HAVE_ACCELERATE) && ! defined(ACCELERATE_NEW_LAPACK)) \
    || defined(BLAS_HAVE_F2C)
    typedef double lapackpp_float_return;
#else
    typedef float lapackpp_float_return;
#endif
// #define so we can check later with #ifdef.
#define lapackpp_float_return lapackpp_float_return

//------------------------------------------------------------------------------
// Complex types
#if defined( lapack_complex_float ) || defined( LAPACK_COMPLEX_CUSTOM )
    // LAPACKE's header may already define lapack_complex_float. Otherwise,
    // if user defines LAPACK_COMPLEX_CUSTOM, then the user must define:
    //     lapack_complex_float
    //     lapack_complex_double

#elif defined(LAPACK_COMPLEX_STRUCTURE) || defined(_MSC_VER)
    // If user defines LAPACK_COMPLEX_STRUCTURE, then use a struct.
    // Also use this for MSVC, as it has no C99 _Complex.
    typedef struct { float  real, imag; } lapack_complex_float;
    typedef struct { double real, imag; } lapack_complex_double;

#elif defined(LAPACK_COMPLEX_CPP) && defined(__cplusplus)
    // If user defines LAPACK_COMPLEX_CPP, then use C++ std::complex.
    // This isn't compatible as a return type from extern C functions,
    // so it may generate compiler warnings or errors.
    #include <complex>
    typedef std::complex<float>  lapack_complex_float;
    typedef std::complex<double> lapack_complex_double;

#else
    // Otherwise, by default use C99 _Complex.
    #include <complex.h>
    typedef float _Complex  lapack_complex_float;
    typedef double _Complex lapack_complex_double;
#endif

// #define so we can check later with #ifdef.
#define lapack_complex_float  lapack_complex_float
#define lapack_complex_double lapack_complex_double

#endif // LAPACK_CONFIG_H
