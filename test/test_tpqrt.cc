#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

#if LAPACK_VERSION >= 30400  // >= 3.4.0

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_tpqrt_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t l = params.l();
    int64_t nb = params.nb();
    int64_t align = params.align();

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();
    params.gflops();

    if (! run)
        return;

    // skip invalid sizes and options
    if (min(m, n) < l || n < nb || nb < 1) {
        printf( "skipping because tpqrt requires min(m, n) >= l and n >= nb >= 1\n" );
        return;
    }

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    int64_t ldb = roundup( max( 1, m ), align );
    int64_t ldt = roundup( nb, align );
    size_t size_A = (size_t) lda * n;  // n-by-n
    size_t size_B = (size_t) ldb * n;  // m-by-n
    size_t size_T = (size_t) ldt * n;  // nb-by-n

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );
    std::vector< scalar_t > T_tst( size_T );
    std::vector< scalar_t > T_ref( size_T );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    A_ref = A_tst;
    B_ref = B_tst;

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = get_wtime();
    int64_t info_tst = lapack::tpqrt( m, n, l, nb, &A_tst[0], lda, &B_tst[0], ldb, &T_tst[0], ldt );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::tpqrt returned error %lld\n", (lld) info_tst );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::geqrf( m, n );  // estimate
    params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_tpqrt( m, n, l, nb, &A_ref[0], lda, &B_ref[0], ldb, &T_ref[0], ldt );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_tpqrt returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( B_tst, B_ref );
        error += abs_error( T_tst, T_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_tpqrt( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_tpqrt_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_tpqrt_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_tpqrt_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_tpqrt_work< std::complex<double> >( params, run );
            break;
    }
}

#else

// -----------------------------------------------------------------------------
void test_tpqrt( Params& params, bool run )
{
    fprintf( stderr, "tpqrt requires LAPACK >= 3.4.0\n\n" );
    exit(0);
}

#endif  // LAPACK >= 3.4.0
