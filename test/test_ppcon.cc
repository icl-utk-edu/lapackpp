#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_ppcon(
    char uplo, lapack_int n, float* AP, float anorm, float* rcond )
{
    return LAPACKE_sppcon( LAPACK_COL_MAJOR, uplo, n, AP, anorm, rcond );
}

static lapack_int LAPACKE_ppcon(
    char uplo, lapack_int n, double* AP, double anorm, double* rcond )
{
    return LAPACKE_dppcon( LAPACK_COL_MAJOR, uplo, n, AP, anorm, rcond );
}

static lapack_int LAPACKE_ppcon(
    char uplo, lapack_int n, std::complex<float>* AP, float anorm, float* rcond )
{
    return LAPACKE_cppcon( LAPACK_COL_MAJOR, uplo, n, AP, anorm, rcond );
}

static lapack_int LAPACKE_ppcon(
    char uplo, lapack_int n, std::complex<double>* AP, double anorm, double* rcond )
{
    return LAPACKE_zppcon( LAPACK_COL_MAJOR, uplo, n, AP, anorm, rcond );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_ppcon_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();

    real_t eps = std::numeric_limits< real_t >::epsilon();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    real_t anorm;
    real_t rcond_tst;
    real_t rcond_ref;
    size_t size_AP = (size_t) (n*(n+1)/2);

    std::vector< scalar_t > AP( size_AP );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AP.size(), &AP[0] );

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

    anorm = lapack::lansp( lapack::Norm::One, uplo, n, &AP[0] );

    // factor A into LL^T
    int64_t info = lapack::pptrf( uplo, n, &AP[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::pptrf returned error %lld\n", (lld) info );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::ppcon( uplo, n, &AP[0], anorm, &rcond_tst );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::ppcon returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::ppcon( n );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_ppcon( uplo2char(uplo), n, &AP[0], anorm, &rcond_ref );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_ppcon returned error %lld\n", (lld) info_ref );
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
        params.okay.value() = (error < 3*eps);
    }
}

// -----------------------------------------------------------------------------
void test_ppcon( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_ppcon_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_ppcon_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_ppcon_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_ppcon_work< std::complex<double> >( params, run );
            break;
    }
}
