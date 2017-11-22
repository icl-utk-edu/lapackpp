#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>
#include <lapacke.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_getrs(
    char trans, lapack_int n, lapack_int nrhs, float* A, lapack_int lda, lapack_int* ipiv, float* B, lapack_int ldb )
{
    return LAPACKE_sgetrs( LAPACK_COL_MAJOR, trans, n, nrhs, A, lda, ipiv, B, ldb );
}

static lapack_int LAPACKE_getrs(
    char trans, lapack_int n, lapack_int nrhs, double* A, lapack_int lda, lapack_int* ipiv, double* B, lapack_int ldb )
{
    return LAPACKE_dgetrs( LAPACK_COL_MAJOR, trans, n, nrhs, A, lda, ipiv, B, ldb );
}

static lapack_int LAPACKE_getrs(
    char trans, lapack_int n, lapack_int nrhs, std::complex<float>* A, lapack_int lda, lapack_int* ipiv, std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cgetrs( LAPACK_COL_MAJOR, trans, n, nrhs, A, lda, ipiv, B, ldb );
}

static lapack_int LAPACKE_getrs(
    char trans, lapack_int n, lapack_int nrhs, std::complex<double>* A, lapack_int lda, lapack_int* ipiv, std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zgetrs( LAPACK_COL_MAJOR, trans, n, nrhs, A, lda, ipiv, B, ldb );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_getrs_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Op trans = params.trans.value();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs.value();
    int64_t align = params.align.value();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    int64_t ldb = roundup( max( 1, n ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_ipiv = (size_t) (n);
    size_t size_B = (size_t) ldb * nrhs;

    std::vector< scalar_t > A( size_A );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A.size(), &A[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    B_ref = B_tst;

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld\n"
                "B n=%5lld, nrhs=%5lld, ldb=%5lld",
                n, lda, n, nrhs, ldb );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( n, n, &A[0], lda );
        printf( "B = " ); print_matrix( n, nrhs, &B_tst[0], lda );
    }

    // factor A into LU
    int64_t info = lapack::getrf( n, n, &A[0], lda, &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::getrf returned error %lld\n", (lld) info );
    }
    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    // test error exits
    if (params.error_exit.value() == 'y') {
        assert_throw( lapack::getrf( -1,  n, &A[0], lda, &ipiv_tst[0] ), lapack::Error );
        assert_throw( lapack::getrf(  n, -1, &A[0], lda, &ipiv_tst[0] ), lapack::Error );
        assert_throw( lapack::getrf(  n,  n, &A[0], n-1, &ipiv_tst[0] ), lapack::Error );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::getrs( trans, n, nrhs, &A[0], lda, &ipiv_tst[0], &B_tst[0], ldb );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::getrs returned error %lld\n", (lld) info_tst );
    }

    double gflop = lapack::Gflop< scalar_t >::getrs( n, nrhs );
    params.time.value()   = time;
    params.gflops.value() = gflop / time;

    if (verbose >= 2) {
        printf( "B2 = " ); print_matrix( n, nrhs, &B_tst[0], ldb );
    }

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_getrs( op2char(trans), n, nrhs, &A[0], lda, &ipiv_ref[0], &B_ref[0], ldb );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_getrs returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value()   = time;
        params.ref_gflops.value() = gflop / time;

        if (verbose >= 2) {
            printf( "B2ref = " ); print_matrix( n, nrhs, &B_ref[0], ldb );
        }

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
void test_getrs( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_getrs_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_getrs_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_getrs_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_getrs_work< std::complex<double> >( params, run );
            break;
    }
}
