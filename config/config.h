// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>

//------------------------------------------------------------------------------
// A Fortran symbol suffix, if defined, is appended to every mangled name, for
// BLAS/LAPACK libraries that suffix their symbols (e.g., OpenBLAS built with
// SYMBOLSUFFIX=64_ exports dgemm_64_, dpstrf_64_). LAPACK_FORTRAN_SUFFIX takes
// precedence; BLAS_FORTRAN_SUFFIX (inherited from BLAS++) is the fallback.
// An explicitly empty LAPACK_FORTRAN_SUFFIX disables suffix inheritance.
#if defined(LAPACK_FORTRAN_SUFFIX)
    #define FORTRAN_SUFFIX_ LAPACK_FORTRAN_SUFFIX
#elif defined(BLAS_FORTRAN_SUFFIX)
    #define FORTRAN_SUFFIX_ BLAS_FORTRAN_SUFFIX
#endif

#ifdef FORTRAN_SUFFIX_
    #define FORTRAN_CONCAT_( a, b ) a##b
    #define FORTRAN_CONCAT(  a, b ) FORTRAN_CONCAT_( a, b )
    #if defined(FORTRAN_UPPER) || defined(BLAS_FORTRAN_UPPER) || defined(LAPACK_FORTRAN_UPPER)
        #define FORTRAN_NAME( lower, UPPER ) FORTRAN_CONCAT( UPPER, FORTRAN_SUFFIX_ )
    #elif defined(FORTRAN_LOWER) || defined(BLAS_FORTRAN_LOWER) || defined(LAPACK_FORTRAN_LOWER)
        #define FORTRAN_NAME( lower, UPPER ) FORTRAN_CONCAT( lower, FORTRAN_SUFFIX_ )
    #else
        // default is ADD_
        #define FORTRAN_NAME( lower, UPPER ) FORTRAN_CONCAT( lower##_, FORTRAN_SUFFIX_ )
    #endif
#else
    #if defined(FORTRAN_UPPER) || defined(BLAS_FORTRAN_UPPER) || defined(LAPACK_FORTRAN_UPPER)
        #define FORTRAN_NAME( lower, UPPER ) UPPER
    #elif defined(FORTRAN_LOWER) || defined(BLAS_FORTRAN_LOWER) || defined(LAPACK_FORTRAN_LOWER)
        #define FORTRAN_NAME( lower, UPPER ) lower
    #elif defined(FORTRAN_ADD_) || defined(BLAS_FORTRAN_ADD_) || defined(LAPACK_FORTRAN_ADD_)
        #define FORTRAN_NAME( lower, UPPER ) lower ## _
    #else
        #error "must define one of FORTRAN_ADD_, FORTRAN_LOWER, FORTRAN_UPPER"
    #endif
#endif

//------------------------------------------------------------------------------
#if defined(BLAS_ILP64) || defined(LAPACK_ILP64)
    // long is >= 32 bits, long long is >= 64 bits
    // macOS Accelerate uses long, Intel MKL uses long long,
    // prefer int64_t (which can be long or long long).
    #ifdef BLAS_HAVE_ACCELERATE
        #define ACCELERATE_LAPACK_ILP64
        typedef long blas_int;
        typedef long lapack_int;
    #else
        typedef int64_t blas_int;
        typedef int64_t lapack_int;
    #endif
#else
    typedef int blas_int;
    typedef int lapack_int;
#endif

//------------------------------------------------------------------------------
#ifdef BLAS_HAVE_ACCELERATE
    // Neither old nor new macOS Accelerate API passes strlen.
    #undef BLAS_FORTRAN_STRLEN_END
    #undef LAPACK_FORTRAN_STRLEN_END
#else
    #ifndef BLAS_FORTRAN_STRLEN_END
    #define BLAS_FORTRAN_STRLEN_END
    #endif

    #ifndef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_FORTRAN_STRLEN_END
    #endif
#endif

#endif // CONFIG_H
