// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_MANGLING_H
#define LAPACK_MANGLING_H

#include "blas/defines.h"
#include "lapack/defines.h"

// -----------------------------------------------------------------------------
// Fortran name mangling depends on compiler.
// Define FORTRAN_UPPER for uppercase,
// define FORTRAN_LOWER for lowercase (IBM xlf),
// define FORTRAN_ADD_  for lowercase with appended underscore
// (GNU gcc, Intel icc, PGI pgfortan, Cray ftn).
//
// A Fortran symbol suffix, if defined, is appended to every mangled name, for
// LAPACK libraries that suffix their symbols, e.g., OpenBLAS built with
// INTERFACE64=1 SYMBOLSUFFIX=64_ exports `dpstrf_64_`. LAPACK_FORTRAN_SUFFIX
// (set via the lapack_symbol_suffix option) takes precedence; BLAS_FORTRAN_SUFFIX
// (inherited from BLAS++'s blas_symbol_suffix via blas/defines.h) is the fallback.
// An explicitly empty LAPACK_FORTRAN_SUFFIX disables suffix inheritance.
#ifndef LAPACK_GLOBAL
    #if defined(LAPACK_FORTRAN_SUFFIX)
        #define LAPACK_GLOBAL_SUFFIX_ LAPACK_FORTRAN_SUFFIX
    #elif defined(BLAS_FORTRAN_SUFFIX)
        #define LAPACK_GLOBAL_SUFFIX_ BLAS_FORTRAN_SUFFIX
    #endif

    #ifdef LAPACK_GLOBAL_SUFFIX_
        #define LAPACK_GLOBAL_CONCAT_( a, b ) a##b
        #define LAPACK_GLOBAL_CONCAT(  a, b ) LAPACK_GLOBAL_CONCAT_( a, b )
        #if defined(BLAS_FORTRAN_UPPER) || defined(LAPACK_FORTRAN_UPPER) || defined(LAPACK_GLOBAL_PATTERN_UC)
            #define LAPACK_GLOBAL( lower, UPPER ) LAPACK_GLOBAL_CONCAT( UPPER, LAPACK_GLOBAL_SUFFIX_ )
        #elif defined(BLAS_FORTRAN_LOWER) || defined(LAPACK_FORTRAN_LOWER) || defined(LAPACK_GLOBAL_PATTERN_LC)
            #define LAPACK_GLOBAL( lower, UPPER ) LAPACK_GLOBAL_CONCAT( lower, LAPACK_GLOBAL_SUFFIX_ )
        #else
            // default is ADD_
            #define LAPACK_GLOBAL( lower, UPPER ) LAPACK_GLOBAL_CONCAT( lower##_, LAPACK_GLOBAL_SUFFIX_ )
        #endif
    #else
        #if defined(BLAS_FORTRAN_UPPER) || defined(LAPACK_FORTRAN_UPPER) || defined(LAPACK_GLOBAL_PATTERN_UC)
            #define LAPACK_GLOBAL( lower, UPPER ) UPPER
        #elif defined(BLAS_FORTRAN_LOWER) || defined(LAPACK_FORTRAN_LOWER) || defined(LAPACK_GLOBAL_PATTERN_LC)
            #define LAPACK_GLOBAL( lower, UPPER ) lower
        #elif defined(BLAS_FORTRAN_ADD_) || defined(LAPACK_FORTRAN_ADD_) || defined(LAPACK_GLOBAL_PATTERN_MC)
            #define LAPACK_GLOBAL( lower, UPPER ) lower##_
        #else
            #error "One of LAPACK_FORTRAN_ADD_, LAPACK_FORTRAN_LOWER, or LAPACK_FORTRAN_UPPER must be defined to set how Fortran functions are name mangled."
        #endif
    #endif
#endif

#endif  /* LAPACK_MANGLING_H */
