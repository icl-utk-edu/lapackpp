#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_ptcon(
    lapack_int n, float* D, float* E, float anorm, float* rcond )
{
    return LAPACKE_sptcon( n, D, E, anorm, rcond );
}

static lapack_int LAPACKE_ptcon(
    lapack_int n, double* D, double* E, double anorm, double* rcond )
{
    return LAPACKE_dptcon( n, D, E, anorm, rcond );
}

static lapack_int LAPACKE_ptcon(
    lapack_int n, float* D, std::complex<float>* E, float anorm, float* rcond )
{
    return LAPACKE_cptcon( n, D, E, anorm, rcond );
}

static lapack_int LAPACKE_ptcon(
    lapack_int n, double* D, std::complex<double>* E, double anorm, double* rcond )
{
    return LAPACKE_zptcon( n, D, E, anorm, rcond );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_ptcon_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    int64_t n = params.dim.n();

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
    size_t size_D = (size_t) (n);
    size_t size_E = (size_t) (n-1);

    std::vector< real_t > D( size_D );
    std::vector< scalar_t > E( size_E );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, D.size(), &D[0] );
    lapack::larnv( idist, iseed, E.size(), &E[0] );
    
    // diagonally dominant -> positive definite
    for (int64_t i = 0; i < n; ++i) {
        D[ i ] += n;
    }

    // factor
    int64_t info = lapack::pttrf( n, &D[0], &E[0] );
    if (info != 0) {
        fprintf( stderr, "LAPACKE_pttrf returned error %lld\n", (lld) info );
    }

    // get Anorm
    anorm = lapack::lanht( lapack::Norm::One, n, &D[0], &E[0] );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::ptcon( n, &D[0], &E[0], anorm, &rcond_tst );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::ptcon returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::ptcon( n );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_ptcon( n, &D[0], &E[0], anorm, &rcond_ref );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_ptcon returned error %lld\n", (lld) info_ref );
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
void test_ptcon( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_ptcon_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_ptcon_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_ptcon_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_ptcon_work< std::complex<double> >( params, run );
            break;
    }
}
