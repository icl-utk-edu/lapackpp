#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_potri(
    char uplo, lapack_int n, float* A, lapack_int lda )
{
    return LAPACKE_spotri( LAPACK_COL_MAJOR, uplo, n, A, lda );
}

static lapack_int LAPACKE_potri(
    char uplo, lapack_int n, double* A, lapack_int lda )
{
    return LAPACKE_dpotri( LAPACK_COL_MAJOR, uplo, n, A, lda );
}

static lapack_int LAPACKE_potri(
    char uplo, lapack_int n, std::complex<float>* A, lapack_int lda )
{
    return LAPACKE_cpotri( LAPACK_COL_MAJOR, uplo, n, A, lda );
}

static lapack_int LAPACKE_potri(
    char uplo, lapack_int n, std::complex<double>* A, lapack_int lda )
{
    return LAPACKE_zpotri( LAPACK_COL_MAJOR, uplo, n, A, lda );
}

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_potrf(
    char uplo, lapack_int n, float* A, lapack_int lda )
{
    return LAPACKE_spotrf( LAPACK_COL_MAJOR, uplo, n, A, lda );
}

static lapack_int LAPACKE_potrf(
    char uplo, lapack_int n, double* A, lapack_int lda )
{
    return LAPACKE_dpotrf( LAPACK_COL_MAJOR, uplo, n, A, lda );
}

static lapack_int LAPACKE_potrf(
    char uplo, lapack_int n, std::complex<float>* A, lapack_int lda )
{
    return LAPACKE_cpotrf( LAPACK_COL_MAJOR, uplo, n, A, lda );
}

static lapack_int LAPACKE_potrf(
    char uplo, lapack_int n, std::complex<double>* A, lapack_int lda )
{
    return LAPACKE_zpotrf( LAPACK_COL_MAJOR, uplo, n, A, lda );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_potri_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();
    params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    size_t size_A = (size_t) lda * n;

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );

    // diagonally dominant -> positive definite
    for (int64_t i = 0; i < n; ++i) {
        A_tst[ i + i*lda ] += n;
    }
    A_ref = A_tst;

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld\n",
                (lld) n, (lld) lda );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( n, n, &A_tst[0], lda );
    }

    // factor A into LL^T
    int64_t info = lapack::potrf( uplo, n, &A_tst[0], lda );
    if (info != 0) {
        fprintf( stderr, "lapack::potrf returned error %lld\n", (lld) info );
    }

    // test error exits
    if (params.error_exit.value() == 'y') {
        assert_throw( lapack::potri( Uplo(0),  n, &A_tst[0], lda ), lapack::Error );
        assert_throw( lapack::potri( uplo,    -1, &A_tst[0], lda ), lapack::Error );
        assert_throw( lapack::potri( uplo,     n, &A_tst[0], n-1 ), lapack::Error );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::potri( uplo, n, &A_tst[0], lda );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::potri returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    double gflop = lapack::Gflop< scalar_t >::potri( n );
    params.gflops.value() = gflop / time;

    if (verbose >= 2) {
        printf( "A2 = " ); print_matrix( n, n, &A_tst[0], lda );
    }

    if (params.check.value() == 'y') {
        // ---------- check error
        // comparing to ref. solution doesn't work due to roundoff errors
        real_t eps = std::numeric_limits< real_t >::epsilon();
        real_t tol = params.tol.value();

        // symmetrize A^{-1}, in order to use hemm
        if (uplo == Uplo::Lower) {
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = 0; i < j; ++i)
                    A_tst[ i + j*lda ] = conj( A_tst[ j + i*lda ] );
        }
        else {
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = 0; i < j; ++i)
                    A_tst[ j + i*lda ] = conj( A_tst[ i + j*lda ] );
        }
        if (verbose >= 2) {
            printf( "A2b = " ); print_matrix( n, n, &A_tst[0], lda );
        }

        // R = I
        std::vector< scalar_t > R( size_A );
        // todo: laset; needs uplo=general
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < n; ++i) {
                R[ i + j*lda ] = 0;
            }
            R[ j + j*lda ] = 1;
        }

        // R = I - A A^{-1}, A is Hermitian, A^{-1} is treated as general
        blas::hemm( Layout::ColMajor, Side::Left, uplo, n, n,
                    -1.0, &A_ref[0], lda,
                          &A_tst[0], lda,
                     1.0, &R[0], lda );
        if (verbose >= 2) {
            printf( "R = " ); print_matrix( n, n, &R[0], lda );
        }

        // error = ||I - A A^{-1}|| / (n ||A|| ||A^{-1}||)
        real_t Rnorm     = lapack::lange( lapack::Norm::Fro, n, n, &R[0], lda );
        real_t Anorm     = lapack::lanhe( lapack::Norm::Fro, uplo, n, &A_ref[0], lda );
        real_t Ainv_norm = lapack::lanhe( lapack::Norm::Fro, uplo, n, &A_tst[0], lda );
        real_t error = Rnorm / (n * Anorm * Ainv_norm);
        params.error.value() = error;
        params.okay.value() = (error < tol*eps);
    }

    if (params.ref.value() == 'y') {
        // factor A into LL^T
        info = LAPACKE_potrf( uplo2char(uplo), n, &A_ref[0], lda );
        if (info != 0) {
            fprintf( stderr, "LAPACKE_potrf returned error %lld\n", (lld) info );
        }

        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_potri( uplo2char(uplo), n, &A_ref[0], lda );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_potri returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        params.ref_gflops.value() = gflop / time;

        if (verbose >= 2) {
            printf( "A2ref = " ); print_matrix( n, n, &A_ref[0], lda );
        }
    }
}

// -----------------------------------------------------------------------------
void test_potri( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_potri_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_potri_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_potri_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_potri_work< std::complex<double> >( params, run );
            break;
    }
}
