#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_ptsv(
    lapack_int n, lapack_int nrhs, float* D, float* E, float* B, lapack_int ldb )
{
    return LAPACKE_sptsv( LAPACK_COL_MAJOR, n, nrhs, D, E, B, ldb );
}

static lapack_int LAPACKE_ptsv(
    lapack_int n, lapack_int nrhs, double* D, double* E, double* B, lapack_int ldb )
{
    return LAPACKE_dptsv( LAPACK_COL_MAJOR, n, nrhs, D, E, B, ldb );
}

static lapack_int LAPACKE_ptsv(
    lapack_int n, lapack_int nrhs, float* D, std::complex<float>* E, std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cptsv( LAPACK_COL_MAJOR, n, nrhs, D, E, B, ldb );
}

static lapack_int LAPACKE_ptsv(
    lapack_int n, lapack_int nrhs, double* D, std::complex<double>* E, std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zptsv( LAPACK_COL_MAJOR, n, nrhs, D, E, B, ldb );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_ptsv_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
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
    size_t size_D = (size_t) (n);
    size_t size_E = (size_t) (n-1);
    size_t size_B = (size_t) ldb * nrhs;

    std::vector< real_t > D_tst( size_D );
    std::vector< real_t > D_ref( size_D );
    std::vector< scalar_t > E_tst( size_E );
    std::vector< scalar_t > E_ref( size_E );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, D_tst.size(), &D_tst[0] );
    lapack::larnv( idist, iseed, E_tst.size(), &E_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    E_ref = E_tst;
    B_ref = B_tst;

    // diagonally dominant -> positive definite
    for (int64_t i = 0; i < n; ++i) {
        D_tst[ i ] += n;
    }
    D_ref = D_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::ptsv( n, nrhs, &D_tst[0], &E_tst[0], &B_tst[0], ldb );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::ptsv returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::ptsv( n, nrhs );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_ptsv( n, nrhs, &D_ref[0], &E_ref[0], &B_ref[0], ldb );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_ptsv returned error %lld\n", (lld) info_ref );
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
        error += abs_error( B_tst, B_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_ptsv( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_ptsv_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_ptsv_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_ptsv_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_ptsv_work< std::complex<double> >( params, run );
            break;
    }
}
