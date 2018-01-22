#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_sptrs(
    char uplo, lapack_int n, lapack_int nrhs, float* AP, lapack_int* ipiv, float* B, lapack_int ldb )
{
    return LAPACKE_ssptrs( LAPACK_COL_MAJOR, uplo, n, nrhs, AP, ipiv, B, ldb );
}

static lapack_int LAPACKE_sptrs(
    char uplo, lapack_int n, lapack_int nrhs, double* AP, lapack_int* ipiv, double* B, lapack_int ldb )
{
    return LAPACKE_dsptrs( LAPACK_COL_MAJOR, uplo, n, nrhs, AP, ipiv, B, ldb );
}

static lapack_int LAPACKE_sptrs(
    char uplo, lapack_int n, lapack_int nrhs, std::complex<float>* AP, lapack_int* ipiv, std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_csptrs( LAPACK_COL_MAJOR, uplo, n, nrhs, AP, ipiv, B, ldb );
}

static lapack_int LAPACKE_sptrs(
    char uplo, lapack_int n, lapack_int nrhs, std::complex<double>* AP, lapack_int* ipiv, std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zsptrs( LAPACK_COL_MAJOR, uplo, n, nrhs, AP, ipiv, B, ldb );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_sptrs_work( Params& params, bool run )
{
    using namespace libtest;
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
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t ldb = roundup( max( 1, n ), align );
    size_t size_AP = (size_t) (n*(n+1)/2);
    size_t size_ipiv = (size_t) (n);
    size_t size_B = (size_t) ldb * nrhs;

    std::vector< scalar_t > AP( size_AP );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AP.size(), &AP[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    B_ref = B_tst;

    // todo: initialize ipiv_tst and ipiv_ref
    int64_t info = lapack::sptrf( uplo, n, &AP[0], &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::sptrf returned error %lld\n", (lld) info );
    }
    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::sptrs( uplo, n, nrhs, &AP[0], &ipiv_tst[0], &B_tst[0], ldb );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::sptrs returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::sptrs( n, nrhs );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_sptrs( uplo2char(uplo), n, nrhs, &AP[0], &ipiv_ref[0], &B_ref[0], ldb );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_sptrs returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

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
void test_sptrs( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_sptrs_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_sptrs_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_sptrs_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_sptrs_work< std::complex<double> >( params, run );
            break;
    }
}