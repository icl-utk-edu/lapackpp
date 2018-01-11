#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_hptri(
    char uplo, lapack_int n, float* AP, lapack_int* ipiv )
{
    return LAPACKE_ssptri( LAPACK_COL_MAJOR, uplo, n, AP, ipiv );
}

static lapack_int LAPACKE_hptri(
    char uplo, lapack_int n, double* AP, lapack_int* ipiv )
{
    return LAPACKE_dsptri( LAPACK_COL_MAJOR, uplo, n, AP, ipiv );
}

static lapack_int LAPACKE_hptri(
    char uplo, lapack_int n, std::complex<float>* AP, lapack_int* ipiv )
{
    return LAPACKE_chptri( LAPACK_COL_MAJOR, uplo, n, AP, ipiv );
}

static lapack_int LAPACKE_hptri(
    char uplo, lapack_int n, std::complex<double>* AP, lapack_int* ipiv )
{
    return LAPACKE_zhptri( LAPACK_COL_MAJOR, uplo, n, AP, ipiv );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_hptri_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    // int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    size_t size_AP = (size_t) (n*(n+1)/2);
    size_t size_ipiv = (size_t) (n);

    std::vector< scalar_t > AP_tst( size_AP );
    std::vector< scalar_t > AP_ref( size_AP );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AP_tst.size(), &AP_tst[0] );

    // initialize ipiv_tst and ipiv_ref
    int64_t info_trf = lapack::hptrf( uplo, n, &AP_tst[0], &ipiv_tst[0] );
    if (info_trf != 0) {
        fprintf( stderr, "lapack::hptrf returned error %lld\n", (lld) info_trf );
    }
    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );
    AP_ref = AP_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::hptri( uplo, n, &AP_tst[0], &ipiv_tst[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::hptri returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::hptri( n );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_hptri( uplo2char(uplo), n, &AP_ref[0], &ipiv_ref[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_hptri returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( AP_tst, AP_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_hptri( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_hptri_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_hptri_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_hptri_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_hptri_work< std::complex<double> >( params, run );
            break;
    }
}
