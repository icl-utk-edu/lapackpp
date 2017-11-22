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
static lapack_int LAPACKE_potrf(
    char uplo, lapack_int n, float* A, lapack_int lda )
{
    return LAPACKE_spotrf( LAPACK_COL_MAJOR, uplo, n, A, lda );
}

static lapack_int LAPACKE_potrf(
    char uplo, lapack_int n, double* A, lapack_int lda )
{
    return LAPACKE_dpotrf( LAPACK_COL_MAJOR, uplo, n, A, lda );
}

static lapack_int LAPACKE_potrf(
    char uplo, lapack_int n, std::complex<float>* A, lapack_int lda )
{
    return LAPACKE_cpotrf( LAPACK_COL_MAJOR, uplo, n, A, lda );
}

static lapack_int LAPACKE_potrf(
    char uplo, lapack_int n, std::complex<double>* A, lapack_int lda )
{
    return LAPACKE_zpotrf( LAPACK_COL_MAJOR, uplo, n, A, lda );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_potrf_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    size_t size_A = (size_t) lda * n;

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    A_ref = A_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::potrf( uplo, n, &A_tst[0], lda );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::potrf returned error %lld\n", (lld) info_tst );
    }

    double gflop = lapack::Gflop< scalar_t >::potrf( n );
    params.time.value()   = time;
    params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_potrf( uplo2char(uplo), n, &A_ref[0], lda );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_potrf returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value()   = time;
        params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_potrf( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_potrf_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_potrf_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_potrf_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_potrf_work< std::complex<double> >( params, run );
            break;
    }
}
