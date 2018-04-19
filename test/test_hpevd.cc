#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_hpevd(
    char jobz, char uplo, lapack_int n, float* AP, float* W, float* Z, lapack_int ldz )
{
    return LAPACKE_sspevd( LAPACK_COL_MAJOR, jobz, uplo, n, AP, W, Z, ldz );
}

static lapack_int LAPACKE_hpevd(
    char jobz, char uplo, lapack_int n, double* AP, double* W, double* Z, lapack_int ldz )
{
    return LAPACKE_dspevd( LAPACK_COL_MAJOR, jobz, uplo, n, AP, W, Z, ldz );
}

static lapack_int LAPACKE_hpevd(
    char jobz, char uplo, lapack_int n, std::complex<float>* AP, float* W, std::complex<float>* Z, lapack_int ldz )
{
    return LAPACKE_chpevd( LAPACK_COL_MAJOR, jobz, uplo, n, AP, W, Z, ldz );
}

static lapack_int LAPACKE_hpevd(
    char jobz, char uplo, lapack_int n, std::complex<double>* AP, double* W, std::complex<double>* Z, lapack_int ldz )
{
    return LAPACKE_zhpevd( LAPACK_COL_MAJOR, jobz, uplo, n, AP, W, Z, ldz );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_hpevd_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
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
    int64_t ldz = roundup( max( 1, n ), align );
    size_t size_AP = (size_t) (n*(n+1)/2);
    size_t size_W = (size_t) (n);
    size_t size_Z = (size_t) ldz * n;

    std::vector< scalar_t > AP_tst( size_AP );
    std::vector< scalar_t > AP_ref( size_AP );
    std::vector< real_t > W_tst( size_W );
    std::vector< real_t > W_ref( size_W );
    std::vector< scalar_t > Z_tst( size_Z );
    std::vector< scalar_t > Z_ref( size_Z );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AP_tst.size(), &AP_tst[0] );
    AP_ref = AP_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::hpevd( jobz, uplo, n, &AP_tst[0], &W_tst[0], &Z_tst[0], ldz );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::hpevd returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::hpevd( jobz, n );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_hpevd( job2char(jobz), uplo2char(uplo), n, &AP_ref[0], &W_ref[0], &Z_ref[0], ldz );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_hpevd returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( AP_tst, AP_ref );
        error += abs_error( W_tst, W_ref );
        error += abs_error( Z_tst, Z_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_hpevd( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_hpevd_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_hpevd_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_hpevd_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_hpevd_work< std::complex<double> >( params, run );
            break;
    }
}
