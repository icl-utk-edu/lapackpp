#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static float LAPACKE_lange(
    char norm, lapack_int m, lapack_int n, float* A, lapack_int lda )
{
    return LAPACKE_slange( LAPACK_COL_MAJOR, norm, m, n, A, lda );
}

static double LAPACKE_lange(
    char norm, lapack_int m, lapack_int n, double* A, lapack_int lda )
{
    return LAPACKE_dlange( LAPACK_COL_MAJOR, norm, m, n, A, lda );
}

static float LAPACKE_lange(
    char norm, lapack_int m, lapack_int n, std::complex<float>* A, lapack_int lda )
{
    return LAPACKE_clange( LAPACK_COL_MAJOR, norm, m, n, A, lda );
}

static double LAPACKE_lange(
    char norm, lapack_int m, lapack_int n, std::complex<double>* A, lapack_int lda )
{
    return LAPACKE_zlange( LAPACK_COL_MAJOR, norm, m, n, A, lda );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_lange_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Norm norm = params.norm.value();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( m, 1 ), align );
    size_t size_A = (size_t) lda * n;

    std::vector< scalar_t > A( size_A );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A.size(), &A[0] );

    if (verbose >= 1) {
        printf( "\n"
                "A m=%5lld, n=%5lld, lda=%5lld\n",
                (lld) m, (lld) n, (lld) lda );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( m, n, &A[0], lda );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    real_t norm_tst = lapack::lange( norm, m, n, &A[0], lda );
    time = omp_get_wtime() - time;

    //double gflop = lapack::Gflop< scalar_t >::lange( norm, m, n );
    params.time.value()   = time;
    //params.gflops.value() = gflop / time;

    if (verbose >= 1) {
        printf( "norm_tst %.8e\n", norm_tst );
    }

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        real_t norm_ref = LAPACKE_lange( norm2char(norm), m, n, &A[0], lda );
        time = omp_get_wtime() - time;

        params.ref_time.value()   = time;
        //params.ref_gflops.value() = gflop / time;

        if (verbose >= 1) {
            printf( "norm_ref %.8e\n", norm_ref );
        }

        // ---------- check error compared to reference
        real_t error = 0;
        error += std::abs( norm_tst - norm_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_lange( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_lange_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_lange_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_lange_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_lange_work< std::complex<double> >( params, run );
            break;
    }
}
