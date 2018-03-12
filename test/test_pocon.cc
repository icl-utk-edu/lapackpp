#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_pocon(
    char uplo, lapack_int n, float* A, lapack_int lda, float anorm, float* rcond )
{
    return LAPACKE_spocon( LAPACK_COL_MAJOR, uplo, n, A, lda, anorm, rcond );
}

static lapack_int LAPACKE_pocon(
    char uplo, lapack_int n, double* A, lapack_int lda, double anorm, double* rcond )
{
    return LAPACKE_dpocon( LAPACK_COL_MAJOR, uplo, n, A, lda, anorm, rcond );
}

static lapack_int LAPACKE_pocon(
    char uplo, lapack_int n, std::complex<float>* A, lapack_int lda, float anorm, float* rcond )
{
    return LAPACKE_cpocon( LAPACK_COL_MAJOR, uplo, n, A, lda, anorm, rcond );
}

static lapack_int LAPACKE_pocon(
    char uplo, lapack_int n, std::complex<double>* A, lapack_int lda, double anorm, double* rcond )
{
    return LAPACKE_zpocon( LAPACK_COL_MAJOR, uplo, n, A, lda, anorm, rcond );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_pocon_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();
    int64_t verbose = params.verbose.value();

    real_t eps = std::numeric_limits< real_t >::epsilon();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    real_t anorm;
    real_t rcond_tst;
    real_t rcond_ref;
    size_t size_A = (size_t) lda * n;

    std::vector< scalar_t > A( size_A );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A.size(), &A[0] );

    // diagonally dominant -> positive definite
    for (int64_t i = 0; i < n; ++i) {
        A[ i + i*lda ] += n;
    }

    anorm = lapack::lanhe( lapack::Norm::One, uplo, n, &A[0], lda );

    if (verbose >= 1) {
        printf( "\n"
                "A n %lld, lda %lld, Anorm %.2e\n",
                (lld) n, (lld) lda, anorm );
    }

    // factor A into LL^T
    int64_t info = lapack::potrf( uplo, n, &A[0], lda );
    if (info != 0) {
        fprintf( stderr, "lapack::potrf returned error %lld\n", (lld) info );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::pocon( uplo, n, &A[0], lda, anorm, &rcond_tst );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::pocon returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::pocon( n );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_pocon( uplo2char(uplo), n, &A[0], lda, anorm, &rcond_ref );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_pocon returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += std::abs( rcond_tst - rcond_ref );
        params.error.value() = error;
        params.okay.value() = (error < 3*eps);
    }
}

// -----------------------------------------------------------------------------
void test_pocon( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_pocon_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_pocon_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_pocon_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_pocon_work< std::complex<double> >( params, run );
            break;
    }
}
