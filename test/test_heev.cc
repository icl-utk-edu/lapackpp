#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_heev(
    char jobz, char uplo, lapack_int n, float* A, lapack_int lda, float* W )
{
    return LAPACKE_ssyev( LAPACK_COL_MAJOR, jobz, uplo, n, A, lda, W );
}

static lapack_int LAPACKE_heev(
    char jobz, char uplo, lapack_int n, double* A, lapack_int lda, double* W )
{
    return LAPACKE_dsyev( LAPACK_COL_MAJOR, jobz, uplo, n, A, lda, W );
}

static lapack_int LAPACKE_heev(
    char jobz, char uplo, lapack_int n, std::complex<float>* A, lapack_int lda, float* W )
{
    return LAPACKE_cheev( LAPACK_COL_MAJOR, jobz, uplo, n, A, lda, W );
}

static lapack_int LAPACKE_heev(
    char jobz, char uplo, lapack_int n, std::complex<double>* A, lapack_int lda, double* W )
{
    return LAPACKE_zheev( LAPACK_COL_MAJOR, jobz, uplo, n, A, lda, W );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_heev_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Job jobz = params.jobz.value();
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_W = (size_t) (n);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< real_t > W_tst( size_W );
    std::vector< real_t > W_ref( size_W );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    A_ref = A_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::heev( jobz, uplo, n, &A_tst[0], lda, &W_tst[0] );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::heev returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::heev( jobz, n );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_heev( job2char(jobz), uplo2char(uplo), n, &A_ref[0], lda, &W_ref[0] );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_heev returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( W_tst, W_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_heev( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_heev_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_heev_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_heev_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_heev_work< std::complex<double> >( params, run );
            break;
    }
}
