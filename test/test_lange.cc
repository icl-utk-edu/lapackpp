#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>
#include <lapacke.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_lange(
    char norm, lapack_int m, lapack_int n, float* A, lapack_int lda )
{
    return LAPACKE_slange( LAPACK_COL_MAJOR, norm, m, n, A, lda );
}

static lapack_int LAPACKE_lange(
    char norm, lapack_int m, lapack_int n, double* A, lapack_int lda )
{
    return LAPACKE_dlange( LAPACK_COL_MAJOR, norm, m, n, A, lda );
}

static lapack_int LAPACKE_lange(
    char norm, lapack_int m, lapack_int n, std::complex<float>* A, lapack_int lda )
{
    return LAPACKE_clange( LAPACK_COL_MAJOR, norm, m, n, A, lda );
}

static lapack_int LAPACKE_lange(
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

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( m, 1 ), align );
    size_t size_A = (size_t) lda * n;

    std::vector< scalar_t > A( size_A );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A.size(), &A[0] );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::lange( norm, m, n, &A[0], lda );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::lange returned error %lld\n", (lld) info_tst );
    }

    double gflop = lapack::Gflop< scalar_t >::lange( norm, m, n );
    params.time.value()   = time;
    params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_lange( norm2char(norm), m, n, &A[0], lda );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_lange returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value()   = time;
        params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
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
