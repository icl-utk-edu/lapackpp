#include <lapack.hh>

#include <vector>
#include <stdio.h>

//------------------------------------------------------------------------------
template <typename T>
void run( int n )
{
    int lda = n;
    std::vector<T> A( lda*n, 1.0 );  // m-by-k

    // Make diagonally dominant to be positive definite.
    for (int i = 0; i < n; ++i)
        A[ i + i*lda ] += n;

    // ... fill in application data into A ...

    // Cholesky factorization of A.
    int info = lapack::potrf( lapack::Uplo::Lower, n, A.data(), lda );
    if (info != 0)
        printf( "potrf error: %d\n", info );
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    int n = 100;
    printf( "run< float >( %d )\n", n );
    run< float  >( n );

    printf( "run< double >( %d )\n", n );
    run< double >( n );

    printf( "run< complex<float> >( %d )\n", n );
    run< std::complex<float>  >( n );

    printf( "run< complex<double> >( %d )\n", n );
    run< std::complex<double> >( n );

    return 0;
}
