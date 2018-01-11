#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_upgtr(
    char uplo, lapack_int n, float* AP, float* tau, float* Q, lapack_int ldq )
{
    return LAPACKE_sopgtr( LAPACK_COL_MAJOR, uplo, n, AP, tau, Q, ldq );
}

static lapack_int LAPACKE_upgtr(
    char uplo, lapack_int n, double* AP, double* tau, double* Q, lapack_int ldq )
{
    return LAPACKE_dopgtr( LAPACK_COL_MAJOR, uplo, n, AP, tau, Q, ldq );
}

static lapack_int LAPACKE_upgtr(
    char uplo, lapack_int n, std::complex<float>* AP, std::complex<float>* tau, std::complex<float>* Q, lapack_int ldq )
{
    return LAPACKE_cupgtr( LAPACK_COL_MAJOR, uplo, n, AP, tau, Q, ldq );
}

static lapack_int LAPACKE_upgtr(
    char uplo, lapack_int n, std::complex<double>* AP, std::complex<double>* tau, std::complex<double>* Q, lapack_int ldq )
{
    return LAPACKE_zupgtr( LAPACK_COL_MAJOR, uplo, n, AP, tau, Q, ldq );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_upgtr_work( Params& params, bool run )
{
    using libtest::get_wtime;
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
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
    int64_t ldq = roundup( max( 1, n ), align );
    size_t size_AP = (size_t) (n*(n+1)/2);
    size_t size_D = (size_t) (n);
    size_t size_E = (size_t) (n-1);
    size_t size_tau = (size_t) (n-1);
    size_t size_Q = (size_t) ldq * n;

    std::vector< scalar_t > AP( size_AP );
    std::vector< scalar_t > tau( size_tau );
    std::vector< scalar_t > Q_tst( size_Q );
    std::vector< scalar_t > Q_ref( size_Q );
    std::vector< real_t > D( size_D );
    std::vector< real_t > E( size_E );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AP.size(), &AP[0] );
    lapack::larnv( idist, iseed, tau.size(), &tau[0] );

    // reduce to tridiagonal form to use the tau later    
    int64_t info = lapack::hptrd( uplo, n, &AP[0], &D[0], &E[0], &tau[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::upgtr returned error %lld\n", (lld) info );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::upgtr( uplo, n, &AP[0], &tau[0], &Q_tst[0], ldq );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::upgtr returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::upgtr( n );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_upgtr( uplo2char(uplo), n, &AP[0], &tau[0], &Q_ref[0], ldq );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_upgtr returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( Q_tst, Q_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_upgtr( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_upgtr_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_upgtr_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_upgtr_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_upgtr_work< std::complex<double> >( params, run );
            break;
    }
}
