// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_CONFIG_H
#define LAPACK_CONFIG_H

#include <stdlib.h>

#if defined(BLAS_ILP64) && ! defined(LAPACK_ILP64)
    #define LAPACK_ILP64
#endif

#ifndef lapack_int
    #ifdef LAPACK_ILP64
        #define lapack_int int64_t
    #else
        #define lapack_int int
    #endif
#endif

#ifndef lapack_logical
    #define lapack_logical lapack_int
#endif

/* f2c, hence MacOS Accelerate, returns double instead of float
 * for sdot, slange, clange, etc. */
#if defined(HAVE_MACOS_ACCELERATE) || defined(HAVE_F2C)
    typedef double lapack_float_return;
#else
    typedef float lapack_float_return;
#endif

/* --------------------------------------------------------------------------
 * Complex types */
#if defined(LAPACK_COMPLEX_CUSTOM)
    /* If user defines LAPACK_COMPLEX_CUSTOM, then the user must define:
     *     lapack_complex_float
     *     lapack_complex_double
     *     lapack_complex_float_real(z)
     *     lapack_complex_float_imag(z)
     *     lapack_complex_double_real(z)
     *     lapack_complex_double_imag(z)
     */

#elif defined(LAPACK_COMPLEX_STRUCTURE)

    /* If user defines LAPACK_COMPLEX_STRUCTURE, then use a struct. */
    typedef struct { float  real, imag; } lapack_complex_float;
    typedef struct { double real, imag; } lapack_complex_double;
    #define lapack_complex_float_real(z)  ((z).real)
    #define lapack_complex_float_imag(z)  ((z).imag)
    #define lapack_complex_double_real(z) ((z).real)
    #define lapack_complex_double_imag(z) ((z).imag)

#elif defined(LAPACK_COMPLEX_CPP) && defined(__cplusplus)
    /* If user defines LAPACK_COMPLEX_CPP, then use C++ std::complex.
     * This isn't compatible as a return type from extern C functions,
     * so it may generate compiler warnings or errors. */
    #include <complex>
    typedef std::complex<float>  lapack_complex_float;
    typedef std::complex<double> lapack_complex_double;
    #define lapack_complex_float_real(z)  ((z).real())
    #define lapack_complex_float_imag(z)  ((z).imag())
    #define lapack_complex_double_real(z) ((z).real())
    #define lapack_complex_double_imag(z) ((z).imag())

#else

    /* Otherwise, by default use C99 _Complex. */
    #include <complex.h>
    typedef float _Complex  lapack_complex_float;
    typedef double _Complex lapack_complex_double;
    #define lapack_complex_float_real(z)  (creal(z))
    #define lapack_complex_float_imag(z)  (cimag(z))
    #define lapack_complex_double_real(z) (creal(z))
    #define lapack_complex_double_imag(z) (cimag(z))

#endif

/* define so we can check later with ifdef */
#define lapack_complex_float  lapack_complex_float
#define lapack_complex_double lapack_complex_double

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

lapack_complex_float  lapack_make_complex_float( float re, float im );
lapack_complex_double lapack_make_complex_double( double re, double im );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#ifndef LAPACK_malloc
#define LAPACK_malloc( size )   malloc( size )
#endif

#ifndef LAPACK_free
#define LAPACK_free( p )        free( p )
#endif

#endif /* LAPACK_CONFIG_H */
