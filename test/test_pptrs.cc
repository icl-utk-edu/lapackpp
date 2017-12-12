#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_pptrs(
    char uplo, lapack_int n, lapack_int nrhs, float* AP, float* B, lapack_int ldb )
{
    return LAPACKE_spptrs( LAPACK_COL_MAJOR, uplo, n, nrhs, AP, B, ldb );
}

static lapack_int LAPACKE_pptrs(
    char uplo, lapack_int n, lapack_int nrhs, double* AP, double* B, lapack_int ldb )
{
    return LAPACKE_dpptrs( LAPACK_COL_MAJOR, uplo, n, nrhs, AP, B, ldb );
}

static lapack_int LAPACKE_pptrs(
    char uplo, lapack_int n, lapack_int nrhs, std::complex<float>* AP, std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cpptrs( LAPACK_COL_MAJOR, uplo, n, nrhs, AP, B, ldb );
}

static lapack_int LAPACKE_pptrs(
    char uplo, lapack_int n, lapack_int nrhs, std::complex<double>* AP, std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zpptrs( LAPACK_COL_MAJOR, uplo, n, nrhs, AP, B, ldb );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_pptrs_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs.value();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t ldb = roundup( max( 1, n ), align );
    size_t size_AP = (size_t) (n*(n+1)/2);
    size_t size_B = (size_t) ldb * nrhs;

    std::vector< scalar_t > AP( size_AP );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AP.size(), &AP[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    B_ref = B_tst;

    // diagonally dominant -> positive definite
    if (uplo == lapack::Uplo::Upper) {
        for (int64_t i = 0; i < n; ++i) {
            AP[ i + 0.5*(i+1)*i ] += n;
        }
    }
    else { // lower
        for (int64_t i = 0; i < n; ++i) {
            AP[ i + n*i - 0.5*i*(i+1) ] += n;
        }
    }

    // factor A into LL^T
    int64_t info = lapack::pptrf( uplo, n, &AP[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::pptrf returned error %lld\n", (lld) info );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::pptrs( uplo, n, nrhs, &AP[0], &B_tst[0], ldb );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::pptrs returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::pptrs( n, nrhs );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_pptrs( uplo2char(uplo), n, nrhs, &AP[0], &B_ref[0], ldb );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_pptrs returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( B_tst, B_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_pptrs( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_pptrs_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_pptrs_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_pptrs_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_pptrs_work< std::complex<double> >( params, run );
            break;
    }
}
