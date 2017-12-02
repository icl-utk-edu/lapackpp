#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gecon(
    char norm, lapack_int n, float* A, lapack_int lda, float anorm, float* rcond )
{
    return LAPACKE_sgecon( LAPACK_COL_MAJOR, norm, n, A, lda, anorm, rcond );
}

static lapack_int LAPACKE_gecon(
    char norm, lapack_int n, double* A, lapack_int lda, double anorm, double* rcond )
{
    return LAPACKE_dgecon( LAPACK_COL_MAJOR, norm, n, A, lda, anorm, rcond );
}

static lapack_int LAPACKE_gecon(
    char norm, lapack_int n, std::complex<float>* A, lapack_int lda, float anorm, float* rcond )
{
    return LAPACKE_cgecon( LAPACK_COL_MAJOR, norm, n, A, lda, anorm, rcond );
}

static lapack_int LAPACKE_gecon(
    char norm, lapack_int n, std::complex<double>* A, lapack_int lda, double anorm, double* rcond )
{
    return LAPACKE_zgecon( LAPACK_COL_MAJOR, norm, n, A, lda, anorm, rcond );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gecon_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Norm norm = params.norm.value();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    real_t anorm;
    real_t rcond_tst;
    real_t rcond_ref;
    size_t size_A = (size_t) lda * n;

    std::vector< scalar_t > A( size_A );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A.size(), &A[0] );

    anorm = lapack::lange( norm, n, n, &A[0], lda );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::gecon( norm, n, &A[0], lda, anorm, &rcond_tst );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gecon returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::gecon( norm, n );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_gecon( norm2char(norm), n, &A[0], lda, anorm, &rcond_ref );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gecon returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += std::abs( rcond_tst - rcond_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gecon( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gecon_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gecon_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gecon_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gecon_work< std::complex<double> >( params, run );
            break;
    }
}
