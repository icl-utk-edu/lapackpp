#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gtsv(
    lapack_int n, lapack_int nrhs, float* DL, float* D, float* DU, float* B, lapack_int ldb )
{
    return LAPACKE_sgtsv( LAPACK_COL_MAJOR, n, nrhs, DL, D, DU, B, ldb );
}

static lapack_int LAPACKE_gtsv(
    lapack_int n, lapack_int nrhs, double* DL, double* D, double* DU, double* B, lapack_int ldb )
{
    return LAPACKE_dgtsv( LAPACK_COL_MAJOR, n, nrhs, DL, D, DU, B, ldb );
}

static lapack_int LAPACKE_gtsv(
    lapack_int n, lapack_int nrhs, std::complex<float>* DL, std::complex<float>* D, std::complex<float>* DU, std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cgtsv( LAPACK_COL_MAJOR, n, nrhs, DL, D, DU, B, ldb );
}

static lapack_int LAPACKE_gtsv(
    lapack_int n, lapack_int nrhs, std::complex<double>* DL, std::complex<double>* D, std::complex<double>* DU, std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zgtsv( LAPACK_COL_MAJOR, n, nrhs, DL, D, DU, B, ldb );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gtsv_work( Params& params, bool run )
{
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
    size_t size_DL = (size_t) (n-1);
    size_t size_D = (size_t) (n);
    size_t size_DU = (size_t) (n-1);
    size_t size_B = (size_t) ldb * nrhs;

    std::vector< scalar_t > DL_tst( size_DL );
    std::vector< scalar_t > DL_ref( size_DL );
    std::vector< scalar_t > D_tst( size_D );
    std::vector< scalar_t > D_ref( size_D );
    std::vector< scalar_t > DU_tst( size_DU );
    std::vector< scalar_t > DU_ref( size_DU );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, DL_tst.size(), &DL_tst[0] );
    lapack::larnv( idist, iseed, D_tst.size(), &D_tst[0] );
    lapack::larnv( idist, iseed, DU_tst.size(), &DU_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    DL_ref = DL_tst;
    D_ref = D_tst;
    DU_ref = DU_tst;
    B_ref = B_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::gtsv( n, nrhs, &DL_tst[0], &D_tst[0], &DU_tst[0], &B_tst[0], ldb );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gtsv returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::gtsv( n, nrhs );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_gtsv( n, nrhs, &DL_ref[0], &D_ref[0], &DU_ref[0], &B_ref[0], ldb );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gtsv returned error %lld\n", (lld) info_ref );
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
        error += abs_error( B_tst, B_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gtsv( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gtsv_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gtsv_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gtsv_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gtsv_work< std::complex<double> >( params, run );
            break;
    }
}
