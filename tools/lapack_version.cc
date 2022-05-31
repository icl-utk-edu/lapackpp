// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>

#include "lapack_config.h"
#include "lapack_mangling.h"

#ifndef LAPACK_ilaver
#define LAPACK_ilaver LAPACK_GLOBAL(ilaver,ILAVER)
extern "C"
void LAPACK_ilaver( lapack_int* major, lapack_int* minor, lapack_int* patch );
#endif

int main( int argc, char** argv )
{
    lapack_int major, minor, patch;
    LAPACK_ilaver( &major, &minor, &patch );
    printf( "LAPACK_VERSION=%d%02d%02d\n", major, minor, patch );
    return 0;
}
