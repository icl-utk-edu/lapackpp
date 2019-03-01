#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_getrf_work( Params& params, bool run )
{
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();
    params.gflops();
    params.matrix.mark();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, m ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_ipiv = (size_t) (min(m,n));

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );

    lapack::generate_matrix( params.matrix, m, n, &A_tst[0], lda );
    A_ref = A_tst;

    if (verbose >= 1) {
        printf( "\n"
                "A m=%5lld, n=%5lld, lda=%5lld\n",
                (lld) m, (lld) n, (lld) lda );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( m, n, &A_tst[0], lda );
    }

    // test error exits
    if (params.error_exit() == 'y') {
        assert_throw( lapack::getrf( -1,  n, &A_tst[0], lda, &ipiv_tst[0] ), lapack::Error );
        assert_throw( lapack::getrf(  m, -1, &A_tst[0], lda, &ipiv_tst[0] ), lapack::Error );
        assert_throw( lapack::getrf(  m,  n, &A_tst[0], m-1, &ipiv_tst[0] ), lapack::Error );
    }

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = libtest::get_wtime();
    int64_t info_tst = lapack::getrf( m, n, &A_tst[0], lda, &ipiv_tst[0] );
    time = libtest::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::getrf returned error %lld\n", (lld) info_tst );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::getrf( m, n );
    params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "A2 = " ); print_matrix( m, n, &A_tst[0], lda );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = libtest::get_wtime();
        int64_t info_ref = LAPACKE_getrf( m, n, &A_ref[0], lda, &ipiv_ref[0] );
        time = libtest::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_getrf returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "A2ref = " ); print_matrix( m, n, &A_ref[0], lda );
        }

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( ipiv_tst, ipiv_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_getrf( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_getrf_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_getrf_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_getrf_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_getrf_work< std::complex<double> >( params, run );
            break;
    }
}
