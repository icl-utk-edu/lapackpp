#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_ungtr(
    char uplo, lapack_int n, float* A, lapack_int lda, float* tau )
{
    return LAPACKE_sorgtr( LAPACK_COL_MAJOR, uplo, n, A, lda, tau );
}

static lapack_int LAPACKE_ungtr(
    char uplo, lapack_int n, double* A, lapack_int lda, double* tau )
{
    return LAPACKE_dorgtr( LAPACK_COL_MAJOR, uplo, n, A, lda, tau );
}

static lapack_int LAPACKE_ungtr(
    char uplo, lapack_int n, std::complex<float>* A, lapack_int lda, std::complex<float>* tau )
{
    return LAPACKE_cungtr( LAPACK_COL_MAJOR, uplo, n, A, lda, tau );
}

static lapack_int LAPACKE_ungtr(
    char uplo, lapack_int n, std::complex<double>* A, lapack_int lda, std::complex<double>* tau )
{
    return LAPACKE_zungtr( LAPACK_COL_MAJOR, uplo, n, A, lda, tau );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_ungtr_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( n, align );
    size_t size_A = (size_t) lda * n;
    size_t size_tau = (size_t) (n-1);
    size_t size_D = (size_t) (n);
    size_t size_E = (size_t) (n-1);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > tau( size_tau );
    std::vector< real_t > D_tst( size_D );
    std::vector< real_t > E_tst( size_E );

    lapack::generate_matrix( params.matrix, n, n, nullptr, &A_tst[0], lda );

    // reduce to tridiagonal form to use the tau later
    int64_t info = lapack::hetrd( uplo, n, &A_tst[0], lda, &D_tst[0], &E_tst[0], &tau[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::gttrf returned error %lld\n", (lld) info );
    }

    // save A_tst for reference run
    A_ref = A_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::ungtr( uplo, n, &A_tst[0], lda, &tau[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::ungtr returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::ungtr( n );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_ungtr( uplo2char(uplo), n, &A_ref[0], lda, &tau[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_ungtr returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_ungtr( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_ungtr_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_ungtr_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_ungtr_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_ungtr_work< std::complex<double> >( params, run );
            break;
    }
}
