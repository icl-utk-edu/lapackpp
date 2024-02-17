// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>

#include "config.h"

#ifdef ACCELERATE_NEW_LAPACK
    #pragma message "include Accelerate.h"
    #include <Accelerate/Accelerate.h>
#endif

#ifndef LAPACK_ilaver
#pragma message "Fortran name"
#define LAPACK_ilaver FORTRAN_NAME( ilaver, ILAVER )
#endif

#ifndef ACCELERATE_NEW_LAPACK
    #pragma message "self-defined"
    #ifdef __cplusplus
    extern "C"
    #endif
    void LAPACK_ilaver( lapack_int* major, lapack_int* minor, lapack_int* patch );
#endif

    #pragma message "ready"


int main( int argc, char** argv )
{
    using llong = long long;
    lapack_int major, minor, patch;
    LAPACK_ilaver( &major, &minor, &patch );
    printf( "LAPACK_VERSION=%lld.%02lld.%02lld\n",
            llong( major ), llong( minor ), llong( patch ) );
    return 0;
}
