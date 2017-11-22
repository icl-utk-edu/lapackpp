#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>
#include <lapacke.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_potrs(
    char uplo, lapack_int n, lapack_int nrhs, float* A, lapack_int lda, float* B, lapack_int ldb )
{
    return LAPACKE_spotrs( LAPACK_COL_MAJOR, uplo, n, nrhs, A, lda, B, ldb );
}

static lapack_int LAPACKE_potrs(
    char uplo, lapack_int n, lapack_int nrhs, double* A, lapack_int lda, double* B, lapack_int ldb )
{
    return LAPACKE_dpotrs( LAPACK_COL_MAJOR, uplo, n, nrhs, A, lda, B, ldb );
}

static lapack_int LAPACKE_potrs(
    char uplo, lapack_int n, lapack_int nrhs, std::complex<float>* A, lapack_int lda, std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cpotrs( LAPACK_COL_MAJOR, uplo, n, nrhs, A, lda, B, ldb );
}

static lapack_int LAPACKE_potrs(
    char uplo, lapack_int n, lapack_int nrhs, std::complex<double>* A, lapack_int lda, std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zpotrs( LAPACK_COL_MAJOR, uplo, n, nrhs, A, lda, B, ldb );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_potrs_work( Params& params, bool run )
{
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
    params.ref_gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    int64_t ldb = roundup( max( 1, n ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_B = (size_t) ldb * nrhs;

    std::vector< scalar_t > A( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A.size(), &A[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    B_ref = B_tst;

    // diagonally dominant -> positive definite
    for (int64_t i = 0; i < n; ++i) {
        A[ i + i*lda ] += n;
    }
    // factor A into LL^T
    int64_t info = lapack::potrf( uplo, n, &A[0], lda );
    if (info != 0) {
        fprintf( stderr, "lapack::potrf returned error %lld\n", (lld) info );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::potrs( uplo, n, nrhs, &A[0], lda, &B_tst[0], ldb );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::potrs returned error %lld\n", (lld) info_tst );
    }

    double gflop = lapack::Gflop< scalar_t >::potrs( n, nrhs );
    params.time.value()   = time;
    params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_potrs( uplo2char(uplo), n, nrhs, &A[0], lda, &B_ref[0], ldb );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_potrs returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value()   = time;
        params.ref_gflops.value() = gflop / time;

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
void test_potrs( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_potrs_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_potrs_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_potrs_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_potrs_work< std::complex<double> >( params, run );
            break;
    }
}
