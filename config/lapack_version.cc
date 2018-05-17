#include <stdio.h>

#include "config.h"

#define LAPACK_ilaver FORTRAN_NAME( ilaver, ILAVER )

#ifdef __cplusplus
extern "C"
#endif
void LAPACK_ilaver( lapack_int* major, lapack_int* minor, lapack_int* patch );

int main( int argc, char** argv )
{
    lapack_int major, minor, patch;
    LAPACK_ilaver( &major, &minor, &patch );
    printf( "LAPACK_VERSION=%lld.%lld.%lld\n",
            (long long) major, (long long) minor, (long long) patch );
    return 0;
}
