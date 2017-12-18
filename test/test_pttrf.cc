#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_pttrf(
    lapack_int n, float* D, float* E )
{
    return LAPACKE_spttrf( n, D, E );
}

static lapack_int LAPACKE_pttrf(
    lapack_int n, double* D, double* E )
{
    return LAPACKE_dpttrf( n, D, E );
}

static lapack_int LAPACKE_pttrf(
    lapack_int n, float* D, std::complex<float>* E )
{
    return LAPACKE_cpttrf( n, D, E );
}

static lapack_int LAPACKE_pttrf(
    lapack_int n, double* D, std::complex<double>* E )
{
    return LAPACKE_zpttrf( n, D, E );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_pttrf_work( Params& params, bool run )
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
    size_t size_D = (size_t) (n);
    size_t size_E = (size_t) (n-1);

    std::vector< real_t > D_tst( size_D );
    std::vector< real_t > D_ref( size_D );
    std::vector< scalar_t > E_tst( size_E );
    std::vector< scalar_t > E_ref( size_E );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, D_tst.size(), &D_tst[0] );
    lapack::larnv( idist, iseed, E_tst.size(), &E_tst[0] );
    E_ref = E_tst;

    // diagonally dominant -> positive definite
    for (int64_t i = 0; i < n; ++i) {
        D_tst[ i ] += n;
    }
    D_ref = D_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::pttrf( n, &D_tst[0], &E_tst[0] );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::pttrf returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::pttrf( n );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_pttrf( n, &D_ref[0], &E_ref[0] );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_pttrf returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( D_tst, D_ref );
        error += abs_error( E_tst, E_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_pttrf( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_pttrf_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_pttrf_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_pttrf_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_pttrf_work< std::complex<double> >( params, run );
            break;
    }
}