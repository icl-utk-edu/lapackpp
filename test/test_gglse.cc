#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gglse_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    // TODO int64_t p = params.p();
    int64_t p = params.dim.k();
    int64_t align = params.align();
    params.matrix.mark();
    params.matrixB.mark();

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();
    params.msg();

    if (! run)
        return;

    // skip invalid sizes
    if (! ((0 <= p) && (p <= n) && (n <= m+p))) {
        params.msg() = "skipping: requires 0 <= p <= n <= m+p";
        return;
    }

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, m ), align );
    int64_t ldb = roundup( blas::max( 1, p ), align );
    size_t size_A = (size_t) ( lda * n );
    size_t size_B = (size_t) ( ldb * n );
    size_t size_C = (size_t) (m);
    size_t size_D = (size_t) (p);
    size_t size_X = (size_t) (n);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );
    std::vector< scalar_t > C_tst( size_C );
    std::vector< scalar_t > C_ref( size_C );
    std::vector< scalar_t > D_tst( size_D );
    std::vector< scalar_t > D_ref( size_D );
    std::vector< scalar_t > X_tst( size_X );
    std::vector< scalar_t > X_ref( size_X );

    lapack::generate_matrix( params.matrix,  m, n, &A_tst[0], lda );
    lapack::generate_matrix( params.matrixB, p, n, &B_tst[0], ldb );
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, C_tst.size(), &C_tst[0] );
    lapack::larnv( idist, iseed, D_tst.size(), &D_tst[0] );
    A_ref = A_tst;
    B_ref = B_tst;
    C_ref = C_tst;
    D_ref = D_tst;

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = libtest::get_wtime();
    int64_t info_tst = lapack::gglse( m, n, p, &A_tst[0], lda, &B_tst[0], ldb, &C_tst[0], &D_tst[0], &X_tst[0] );
    time = libtest::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gglse returned error %lld\n", (lld) info_tst );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::gglse( m, n );
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = libtest::get_wtime();
        int64_t info_ref = LAPACKE_gglse( m, n, p, &A_ref[0], lda, &B_ref[0], ldb, &C_ref[0], &D_ref[0], &X_ref[0] );
        time = libtest::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gglse returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( B_tst, B_ref );
        error += abs_error( C_tst, C_ref );
        error += abs_error( D_tst, D_ref );
        error += abs_error( X_tst, X_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gglse( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gglse_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gglse_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gglse_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gglse_work< std::complex<double> >( params, run );
            break;
    }
}
