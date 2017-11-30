#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_larfg(
    lapack_int n, float* alpha, float* X, lapack_int incx, float* tau )
{
    return LAPACKE_slarfg( n, alpha, X, incx, tau );
}

static lapack_int LAPACKE_larfg(
    lapack_int n, double* alpha, double* X, lapack_int incx, double* tau )
{
    return LAPACKE_dlarfg( n, alpha, X, incx, tau );
}

static lapack_int LAPACKE_larfg(
    lapack_int n, std::complex<float>* alpha, std::complex<float>* X, lapack_int incx, std::complex<float>* tau )
{
    return LAPACKE_clarfg( n, alpha, X, incx, tau );
}

static lapack_int LAPACKE_larfg(
    lapack_int n, std::complex<double>* alpha, std::complex<double>* X, lapack_int incx, std::complex<double>* tau )
{
    return LAPACKE_zlarfg( n, alpha, X, incx, tau );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_larfg_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t incx = params.incx.value();
    scalar_t alpha_tst = params.alpha.value();
    scalar_t alpha_ref = alpha_tst;
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();

    if (! run)
        return;

    // ---------- setup
    scalar_t tau_tst;
    scalar_t tau_ref;
    size_t size_X = (size_t) (1+(n-2)*abs(incx));

    std::vector< scalar_t > X_tst( size_X );
    std::vector< scalar_t > X_ref( size_X );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, X_tst.size(), &X_tst[0] );
    X_ref = X_tst;

    if (verbose >= 1) {
        printf( "x incx %lld, size %lld\n", (lld) incx, (lld) size_X );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e\n", real(alpha_tst) );
        printf( "x = " ); print_vector( n-1, &X_tst[0], incx );
        printf( "xref = " ); print_vector( n-1, &X_ref[0], incx );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    lapack::larfg( n, &alpha_tst, &X_tst[0], incx, &tau_tst );
    time = omp_get_wtime() - time;

    double gflop = lapack::Gflop< scalar_t >::larfg( n );
    params.time.value()   = time;
    params.gflops.value() = gflop / time;

    if (verbose >= 2) {
        printf( "alpha2 = %.4e\n", real(alpha_tst) );
        printf( "x2 = " ); print_vector( n-1, &X_tst[0], incx );
        printf( "tau = %.4e\n", real(tau_tst) );
    }

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_larfg( n, &alpha_ref, &X_ref[0], incx, &tau_ref );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            printf( "LAPACKE_larfg returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value()   = time;
        params.ref_gflops.value() = gflop / time;


        if (verbose >= 2) {
            printf( "alpha2ref = %.4e\n", real(alpha_ref) );
            printf( "x2ref = " ); print_vector( n-1, &X_ref[0], incx );
            printf( "tau_ref = %.4e\n", real(tau_ref) );
        }

        // ---------- check error compared to reference
        real_t error = 0;
        error += std::abs( alpha_tst - alpha_ref );
        error += abs_error( X_tst, X_ref );
        error += std::abs( tau_tst - tau_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_larfg( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_larfg_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_larfg_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_larfg_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_larfg_work< std::complex<double> >( params, run );
            break;
    }
}
