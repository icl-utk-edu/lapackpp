#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gglse(
    lapack_int m, lapack_int n, lapack_int p, float* A, lapack_int lda, float* B, lapack_int ldb, float* C, float* D, float* X )
{
    return LAPACKE_sgglse( LAPACK_COL_MAJOR, m, n, p, A, lda, B, ldb, C, D, X );
}

static lapack_int LAPACKE_gglse(
    lapack_int m, lapack_int n, lapack_int p, double* A, lapack_int lda, double* B, lapack_int ldb, double* C, double* D, double* X )
{
    return LAPACKE_dgglse( LAPACK_COL_MAJOR, m, n, p, A, lda, B, ldb, C, D, X );
}

static lapack_int LAPACKE_gglse(
    lapack_int m, lapack_int n, lapack_int p, std::complex<float>* A, lapack_int lda, std::complex<float>* B, lapack_int ldb, std::complex<float>* C, std::complex<float>* D, std::complex<float>* X )
{
    return LAPACKE_cgglse( LAPACK_COL_MAJOR, m, n, p, A, lda, B, ldb, C, D, X );
}

static lapack_int LAPACKE_gglse(
    lapack_int m, lapack_int n, lapack_int p, std::complex<double>* A, lapack_int lda, std::complex<double>* B, lapack_int ldb, std::complex<double>* C, std::complex<double>* D, std::complex<double>* X )
{
    return LAPACKE_zgglse( LAPACK_COL_MAJOR, m, n, p, A, lda, B, ldb, C, D, X );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gglse_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    // TODO int64_t p = params.p.value();
    int64_t p = params.dim.k();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;
    
    if (! ((0 <= p) && (p <= n) && ( n <= m+p ))) {
        printf( "skipping because gglse requires 0 <= p <= n <= m+p\n" );
        return;
    }

    // ---------- setup
    int64_t lda = roundup( max( 1, m ), align );
    int64_t ldb = roundup( max( 1, p ), align );
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

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    lapack::larnv( idist, iseed, C_tst.size(), &C_tst[0] );
    lapack::larnv( idist, iseed, D_tst.size(), &D_tst[0] );
    A_ref = A_tst;
    B_ref = B_tst;
    C_ref = C_tst;
    D_ref = D_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::gglse( m, n, p, &A_tst[0], lda, &B_tst[0], ldb, &C_tst[0], &D_tst[0], &X_tst[0] );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gglse returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::gglse( m, n );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_gglse( m, n, p, &A_ref[0], lda, &B_ref[0], ldb, &C_ref[0], &D_ref[0], &X_ref[0] );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gglse returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

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
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gglse( Params& params, bool run )
{
    switch (params.datatype.value()) {
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
