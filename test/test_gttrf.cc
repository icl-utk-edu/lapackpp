#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gttrf(
    lapack_int n, float* DL, float* D, float* DU, float* DU2, lapack_int* ipiv )
{
    return LAPACKE_sgttrf( n, DL, D, DU, DU2, ipiv );
}

static lapack_int LAPACKE_gttrf(
    lapack_int n, double* DL, double* D, double* DU, double* DU2, lapack_int* ipiv )
{
    return LAPACKE_dgttrf( n, DL, D, DU, DU2, ipiv );
}

static lapack_int LAPACKE_gttrf(
    lapack_int n, std::complex<float>* DL, std::complex<float>* D, std::complex<float>* DU, std::complex<float>* DU2, lapack_int* ipiv )
{
    return LAPACKE_cgttrf( n, DL, D, DU, DU2, ipiv );
}

static lapack_int LAPACKE_gttrf(
    lapack_int n, std::complex<double>* DL, std::complex<double>* D, std::complex<double>* DU, std::complex<double>* DU2, lapack_int* ipiv )
{
    return LAPACKE_zgttrf( n, DL, D, DU, DU2, ipiv );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gttrf_work( Params& params, bool run )
{
    using namespace libtest;
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
    size_t size_DL = (size_t) (n-1);
    size_t size_D = (size_t) (n);
    size_t size_DU = (size_t) (n-1);
    size_t size_DU2 = (size_t) (n-2);
    size_t size_ipiv = (size_t) (n);

    std::vector< scalar_t > DL_tst( size_DL );
    std::vector< scalar_t > DL_ref( size_DL );
    std::vector< scalar_t > D_tst( size_D );
    std::vector< scalar_t > D_ref( size_D );
    std::vector< scalar_t > DU_tst( size_DU );
    std::vector< scalar_t > DU_ref( size_DU );
    std::vector< scalar_t > DU2_tst( size_DU2 );
    std::vector< scalar_t > DU2_ref( size_DU2 );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, DL_tst.size(), &DL_tst[0] );
    lapack::larnv( idist, iseed, D_tst.size(), &D_tst[0] );
    lapack::larnv( idist, iseed, DU_tst.size(), &DU_tst[0] );
    DL_ref = DL_tst;
    D_ref = D_tst;
    DU_ref = DU_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::gttrf( n, &DL_tst[0], &D_tst[0], &DU_tst[0], &DU2_tst[0], &ipiv_tst[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gttrf returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::gttrf( n );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_gttrf( n, &DL_ref[0], &D_ref[0], &DU_ref[0], &DU2_ref[0], &ipiv_ref[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gttrf returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( DL_tst, DL_ref );
        error += abs_error( D_tst, D_ref );
        error += abs_error( DU_tst, DU_ref );
        error += abs_error( DU2_tst, DU2_ref );
        error += abs_error( ipiv_tst, ipiv_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gttrf( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gttrf_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gttrf_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gttrf_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gttrf_work< std::complex<double> >( params, run );
            break;
    }
}
