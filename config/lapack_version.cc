// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>

#include "config.h"

#define LAPACK_ilaver FORTRAN_NAME( ilaver, ILAVER )

#ifdef __cplusplus
extern "C"
#endif
void LAPACK_ilaver( lapack_int* major, lapack_int* minor, lapack_int* patch );

int main( int argc, char** argv )
{
    using llong = long long;
    lapack_int major, minor, patch;
    LAPACK_ilaver( &major, &minor, &patch );
    printf( "LAPACK_VERSION=%lld.%02lld.%02lld\n",
            llong( major ), llong( minor ), llong( patch ) );
    return 0;
}
