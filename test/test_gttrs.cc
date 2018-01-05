#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gttrs(
    char trans, lapack_int n, lapack_int nrhs, float* DL, float* D, float* DU, float* DU2, lapack_int* ipiv, float* B, lapack_int ldb )
{
    return LAPACKE_sgttrs( LAPACK_COL_MAJOR, trans, n, nrhs, DL, D, DU, DU2, ipiv, B, ldb );
}

static lapack_int LAPACKE_gttrs(
    char trans, lapack_int n, lapack_int nrhs, double* DL, double* D, double* DU, double* DU2, lapack_int* ipiv, double* B, lapack_int ldb )
{
    return LAPACKE_dgttrs( LAPACK_COL_MAJOR, trans, n, nrhs, DL, D, DU, DU2, ipiv, B, ldb );
}

static lapack_int LAPACKE_gttrs(
    char trans, lapack_int n, lapack_int nrhs, std::complex<float>* DL, std::complex<float>* D, std::complex<float>* DU, std::complex<float>* DU2, lapack_int* ipiv, std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cgttrs( LAPACK_COL_MAJOR, trans, n, nrhs, DL, D, DU, DU2, ipiv, B, ldb );
}

static lapack_int LAPACKE_gttrs(
    char trans, lapack_int n, lapack_int nrhs, std::complex<double>* DL, std::complex<double>* D, std::complex<double>* DU, std::complex<double>* DU2, lapack_int* ipiv, std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zgttrs( LAPACK_COL_MAJOR, trans, n, nrhs, DL, D, DU, DU2, ipiv, B, ldb );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gttrs_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Op trans = params.trans.value();
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
    size_t size_DU2 = (size_t) (n-2);
    size_t size_ipiv = (size_t) (n);
    size_t size_B = (size_t) ldb * nrhs;

    std::vector< scalar_t > DL( size_DL );
    std::vector< scalar_t > D( size_D );
    std::vector< scalar_t > DU( size_DU );
    std::vector< scalar_t > DU2( size_DU2 );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, DL.size(), &DL[0] );
    lapack::larnv( idist, iseed, D.size(), &D[0] );
    lapack::larnv( idist, iseed, DU.size(), &DU[0] );
    lapack::larnv( idist, iseed, DU2.size(), &DU2[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    B_ref = B_tst;

    // factor
    int64_t info = lapack::gttrf( n, &DL[0], &D[0], &DU[0], &DU2[0], &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::gttrf returned error %lld\n", (lld) info );
    }

    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::gttrs( trans, n, nrhs, &DL[0], &D[0], &DU[0], &DU2[0], &ipiv_tst[0], &B_tst[0], ldb );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gttrs returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::gttrs( trans, n, nrhs );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_gttrs( op2char(trans), n, nrhs, &DL[0], &D[0], &DU[0], &DU2[0], &ipiv_ref[0], &B_ref[0], ldb );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gttrs returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( B_tst, B_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gttrs( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gttrs_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gttrs_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gttrs_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gttrs_work< std::complex<double> >( params, run );
            break;
    }
}
