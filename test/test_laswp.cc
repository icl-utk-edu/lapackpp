#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_laswp(
    lapack_int n, float* A, lapack_int lda, lapack_int k1, lapack_int k2, lapack_int* ipiv, lapack_int incx )
{
    return LAPACKE_slaswp( LAPACK_COL_MAJOR, n, A, lda, k1, k2, ipiv, incx );
}

static lapack_int LAPACKE_laswp(
    lapack_int n, double* A, lapack_int lda, lapack_int k1, lapack_int k2, lapack_int* ipiv, lapack_int incx )
{
    return LAPACKE_dlaswp( LAPACK_COL_MAJOR, n, A, lda, k1, k2, ipiv, incx );
}

static lapack_int LAPACKE_laswp(
    lapack_int n, std::complex<float>* A, lapack_int lda, lapack_int k1, lapack_int k2, lapack_int* ipiv, lapack_int incx )
{
    return LAPACKE_claswp( LAPACK_COL_MAJOR, n, A, lda, k1, k2, ipiv, incx );
}

static lapack_int LAPACKE_laswp(
    lapack_int n, std::complex<double>* A, lapack_int lda, lapack_int k1, lapack_int k2, lapack_int* ipiv, lapack_int incx )
{
    return LAPACKE_zlaswp( LAPACK_COL_MAJOR, n, A, lda, k1, k2, ipiv, incx );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_laswp_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t incx = params.incx.value();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t nb = min( 32, n );
    int64_t lda = roundup( max( 1, m ), align );
    int64_t k1 = 1;
    int64_t k2 = nb;
    size_t size_A = (size_t) lda * n;
    size_t size_ipiv = (size_t) (k1+(k2-k1)*abs(incx));

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );

    // factor first panel of A, to get ipiv
    int64_t info = lapack::getrf( m, nb, &A_tst[0], lda, &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::getrf returned error %lld\n", (lld) info );
    }
    A_ref = A_tst;
    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    lapack::laswp( n, &A_tst[0], lda, k1, k2, &ipiv_tst[0], incx );
    time = get_wtime() - time;

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::laswp( n );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_laswp( n, &A_ref[0], lda, k1, k2, &ipiv_ref[0], incx );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_laswp returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        error += abs_error( A_tst, A_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_laswp( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_laswp_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_laswp_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_laswp_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_laswp_work< std::complex<double> >( params, run );
            break;
    }
}
