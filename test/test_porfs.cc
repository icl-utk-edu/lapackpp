#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_porfs(
    char uplo, lapack_int n, lapack_int nrhs, float* A, lapack_int lda, float* AF, lapack_int ldaf, float* B, lapack_int ldb, float* X, lapack_int ldx, float* ferr, float* berr )
{
    return LAPACKE_sporfs( LAPACK_COL_MAJOR, uplo, n, nrhs, A, lda, AF, ldaf, B, ldb, X, ldx, ferr, berr );
}

static lapack_int LAPACKE_porfs(
    char uplo, lapack_int n, lapack_int nrhs, double* A, lapack_int lda, double* AF, lapack_int ldaf, double* B, lapack_int ldb, double* X, lapack_int ldx, double* ferr, double* berr )
{
    return LAPACKE_dporfs( LAPACK_COL_MAJOR, uplo, n, nrhs, A, lda, AF, ldaf, B, ldb, X, ldx, ferr, berr );
}

static lapack_int LAPACKE_porfs(
    char uplo, lapack_int n, lapack_int nrhs, std::complex<float>* A, lapack_int lda, std::complex<float>* AF, lapack_int ldaf, std::complex<float>* B, lapack_int ldb, std::complex<float>* X, lapack_int ldx, float* ferr, float* berr )
{
    return LAPACKE_cporfs( LAPACK_COL_MAJOR, uplo, n, nrhs, A, lda, AF, ldaf, B, ldb, X, ldx, ferr, berr );
}

static lapack_int LAPACKE_porfs(
    char uplo, lapack_int n, lapack_int nrhs, std::complex<double>* A, lapack_int lda, std::complex<double>* AF, lapack_int ldaf, std::complex<double>* B, lapack_int ldb, std::complex<double>* X, lapack_int ldx, double* ferr, double* berr )
{
    return LAPACKE_zporfs( LAPACK_COL_MAJOR, uplo, n, nrhs, A, lda, AF, ldaf, B, ldb, X, ldx, ferr, berr );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_porfs_work( Params& params, bool run )
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

    real_t eps = std::numeric_limits< real_t >::epsilon();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    // make A and AF, B and X, the same size
    int64_t lda = roundup( max( 1, n ), align );
    int64_t ldaf = lda;
    int64_t ldb = roundup( max( 1, n ), align );
    int64_t ldx = ldb;
    size_t size_A = (size_t) lda * n;
    size_t size_AF = (size_t) ldaf * n;
    size_t size_B = (size_t) ldb * nrhs;
    size_t size_X = (size_t) ldx * nrhs;
    size_t size_ferr = (size_t) (nrhs);
    size_t size_berr = (size_t) (nrhs);

    std::vector< scalar_t > A( size_A );
    std::vector< scalar_t > AF( size_AF );
    std::vector< scalar_t > B( size_B );
    std::vector< scalar_t > X_tst( size_X );
    std::vector< scalar_t > X_ref( size_X );
    std::vector< real_t > ferr_tst( size_ferr );
    std::vector< real_t > ferr_ref( size_ferr );
    std::vector< real_t > berr_tst( size_berr );
    std::vector< real_t > berr_ref( size_berr );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A.size(), &A[0] );
    lapack::larnv( idist, iseed, B.size(), &B[0] );

    // diagonally dominant -> positive definite
    for (int64_t i = 0; i < n; ++i) {
        A[ i + i*lda ] += n;
    }

    // factor AF = LU
    AF = A;
    X_tst = B;
    int64_t info = lapack::potrf( uplo, n, &AF[0], lda );
    if (info != 0) {
        fprintf( stderr, "lapack::potrf returned error %lld\n", (lld) info );
    }

    // initial solve of AF X = B
    info = lapack::potrs( uplo, n, nrhs, &AF[0], lda, &X_tst[0], ldx );
    if (info != 0) {
        fprintf( stderr, "lapack::potrs returned error %lld\n", (lld) info );
    }
    X_ref = X_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::porfs( uplo, n, nrhs, &A[0], lda, &AF[0], ldaf, &B[0], ldb, &X_tst[0], ldx, &ferr_tst[0], &berr_tst[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::porfs returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::porfs( n, nrhs );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_porfs( uplo2char(uplo), n, nrhs, &A[0], lda, &AF[0], ldaf, &B[0], ldb, &X_ref[0], ldx, &ferr_ref[0], &berr_ref[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_porfs returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( X_tst, X_ref );
        error += abs_error( ferr_tst, ferr_ref );
        error += abs_error( berr_tst, berr_ref );
        params.error.value() = error;
        params.okay.value() = (error < 3*eps);
    }
}

// -----------------------------------------------------------------------------
void test_porfs( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_porfs_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_porfs_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_porfs_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_porfs_work< std::complex<double> >( params, run );
            break;
    }
}
