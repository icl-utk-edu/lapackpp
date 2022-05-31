// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
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
#ifndef LAPACK_GLOBAL
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

#endif  /* LAPACK_MANGLING_H */
