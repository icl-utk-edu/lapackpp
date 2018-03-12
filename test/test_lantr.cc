#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static float LAPACKE_lantr(
    char norm, char uplo, char diag, lapack_int m, lapack_int n, float* A, lapack_int lda )
{
    return LAPACKE_slantr( LAPACK_COL_MAJOR, norm, uplo, diag, m, n, A, lda );
}

static double LAPACKE_lantr(
    char norm, char uplo, char diag, lapack_int m, lapack_int n, double* A, lapack_int lda )
{
    return LAPACKE_dlantr( LAPACK_COL_MAJOR, norm, uplo, diag, m, n, A, lda );
}

static float LAPACKE_lantr(
    char norm, char uplo, char diag, lapack_int m, lapack_int n, std::complex<float>* A, lapack_int lda )
{
    return LAPACKE_clantr( LAPACK_COL_MAJOR, norm, uplo, diag, m, n, A, lda );
}

static double LAPACKE_lantr(
    char norm, char uplo, char diag, lapack_int m, lapack_int n, std::complex<double>* A, lapack_int lda )
{
    return LAPACKE_zlantr( LAPACK_COL_MAJOR, norm, uplo, diag, m, n, A, lda );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_lantr_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Norm norm = params.norm.value();
    lapack::Uplo uplo = params.uplo.value();
    lapack::Diag diag = params.diag.value();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( m, 1 ), align );
    size_t size_A = (size_t) lda * n;

    std::vector< scalar_t > A( size_A );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A.size(), &A[0] );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    real_t norm_tst = lapack::lantr( norm, uplo, diag, m, n, &A[0], lda );
    time = get_wtime() - time;

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::lantr( norm, diag, m, n );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        real_t norm_ref = LAPACKE_lantr( norm2char(norm), uplo2char(uplo), diag2char(diag), m, n, &A[0], lda );
        time = get_wtime() - time;

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        error += std::abs( norm_tst - norm_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_lantr( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_lantr_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_lantr_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_lantr_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_lantr_work< std::complex<double> >( params, run );
            break;
    }
}
