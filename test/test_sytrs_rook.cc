#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

#if LAPACK_VERSION_MAJOR >= 3 && LAPACK_VERSION_MINOR >= 5  // >= 3.5

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_sytrs_rook(
    char uplo, lapack_int n, lapack_int nrhs, float* A, lapack_int lda, lapack_int* ipiv, float* B, lapack_int ldb )
{
    return LAPACKE_ssytrs_rook( LAPACK_COL_MAJOR, uplo, n, nrhs, A, lda, ipiv, B, ldb );
}

static lapack_int LAPACKE_sytrs_rook(
    char uplo, lapack_int n, lapack_int nrhs, double* A, lapack_int lda, lapack_int* ipiv, double* B, lapack_int ldb )
{
    return LAPACKE_dsytrs_rook( LAPACK_COL_MAJOR, uplo, n, nrhs, A, lda, ipiv, B, ldb );
}

static lapack_int LAPACKE_sytrs_rook(
    char uplo, lapack_int n, lapack_int nrhs, std::complex<float>* A, lapack_int lda, lapack_int* ipiv, std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_csytrs_rook( LAPACK_COL_MAJOR, uplo, n, nrhs, A, lda, ipiv, B, ldb );
}

static lapack_int LAPACKE_sytrs_rook(
    char uplo, lapack_int n, lapack_int nrhs, std::complex<double>* A, lapack_int lda, lapack_int* ipiv, std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zsytrs_rook( LAPACK_COL_MAJOR, uplo, n, nrhs, A, lda, ipiv, B, ldb );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_sytrs_rook_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs.value();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    int64_t ldb = roundup( max( 1, n ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_ipiv = (size_t) (n);
    size_t size_B = (size_t) ldb * nrhs;

    std::vector< scalar_t > A( size_A );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A.size(), &A[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    B_ref = B_tst;

    // factor A
    int64_t info_trf = lapack::sytrf_rook( uplo, n, &A[0], lda, &ipiv_tst[0] );
    if (info_trf != 0) {
        fprintf( stderr, "lapack::sytrf_rook returned error %lld\n", (lld) info_trf );
    }
    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::sytrs_rook( uplo, n, nrhs, &A[0], lda, &ipiv_tst[0], &B_tst[0], ldb );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::sytrs_rook returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::sytrs_rook( n, nrhs );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_sytrs_rook( uplo2char(uplo), n, nrhs, &A[0], lda, &ipiv_ref[0], &B_ref[0], ldb );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_sytrs_rook returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( B_tst, B_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_sytrs_rook( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_sytrs_rook_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_sytrs_rook_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_sytrs_rook_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_sytrs_rook_work< std::complex<double> >( params, run );
            break;
    }
}

#else

// -----------------------------------------------------------------------------
void test_sytrs_rook( Params& params, bool run )
{
    fprintf( stderr, "sytrs_rook requires LAPACK >= 3.5\n\n" );
    exit(0);
}

#endif  // LAPACK >= 3.5
