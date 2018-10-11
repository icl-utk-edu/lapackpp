#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

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
    int64_t incx = params.incx();
    int64_t align = params.align();
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t nb = min( 32, n );
    int64_t lda = roundup( max( 1, m ), align );
    int64_t k1 = 1;
    int64_t k2 = nb;
    size_t size_A = (size_t) lda * n;
    size_t size_ipiv = (size_t) (k1+(k2-k1)*std::abs(incx));

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );

    lapack::generate_matrix( params.matrix, m, n, &A_tst[0], lda );

    // factor first panel of A, to get ipiv
    int64_t info = lapack::getrf( m, nb, &A_tst[0], lda, &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::getrf returned error %lld\n", (lld) info );
    }
    A_ref = A_tst;
    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = get_wtime();
    lapack::laswp( n, &A_tst[0], lda, k1, k2, &ipiv_tst[0], incx );
    time = get_wtime() - time;

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::laswp( n );
    //params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_laswp( n, &A_ref[0], lda, k1, k2, &ipiv_ref[0], incx );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_laswp returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        error += abs_error( A_tst, A_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_laswp( Params& params, bool run )
{
    switch (params.datatype()) {
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
