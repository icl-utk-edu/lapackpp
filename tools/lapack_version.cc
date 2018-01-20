#include <stdio.h>

#ifdef __cplusplus
extern "C"
#endif
void ilaver_( int* major, int* minor, int* patch );

int main( int argc, char** argv )
{
    int major, minor, patch;
    ilaver_( &major, &minor, &patch );
    printf( "LAPACK_VERSION=%d%02d%02d\n", major, minor, patch );
}
