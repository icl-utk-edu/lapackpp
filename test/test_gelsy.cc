#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gelsy_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t align = params.align();
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, m ), align );
    int64_t ldb = roundup( max( max( 1, m), n ), align );
    real_t rcond;
    int64_t rank_tst;
    lapack_int rank_ref;
    size_t size_A = (size_t) lda * n;
    size_t size_B = (size_t) ldb * nrhs;
    size_t size_jpvt = (size_t) (n);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );
    std::vector< int64_t > jpvt_tst( size_jpvt );
    std::vector< lapack_int > jpvt_ref( size_jpvt );

    lapack::generate_matrix( params.matrix, m, n, &A_tst[0], lda );
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );

    A_ref = A_tst;
    B_ref = B_tst;

    // TODO: Initializing jpvt[i] at i
    for (int64_t i = 0; i < n; ++i)
        jpvt_tst[i] = i;
    std::copy( jpvt_tst.begin(), jpvt_tst.end(), jpvt_ref.begin() );

    // TODO: rcond value is set to a meaningless value, fix this
    rcond = 0;

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = get_wtime();
    int64_t info_tst = lapack::gelsy( m, n, nrhs, &A_tst[0], lda, &B_tst[0], ldb, &jpvt_tst[0], rcond, &rank_tst );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gelsy returned error %lld\n", (lld) info_tst );
    }

    // double gflop = lapack::Gflop< scalar_t >::gelsy( m, n, nrhs );
    params.time()   = time;
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_gelsy( m, n, nrhs, &A_ref[0], lda, &B_ref[0], ldb, &jpvt_ref[0], rcond, &rank_ref );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gelsy returned error %lld\n", (lld) info_ref );
        }

        params.ref_time()   = time;
        // params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        real_t eps = std::numeric_limits< real_t >::epsilon();
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( B_tst, B_ref );
        error += abs_error( jpvt_tst, jpvt_ref );
        error += std::abs( rank_tst - rank_ref );
        params.error() = error;
        params.okay() = (error < 3*eps);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gelsy( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gelsy_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gelsy_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gelsy_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gelsy_work< std::complex<double> >( params, run );
            break;
    }
}
