#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_hegvd(
    lapack_int itype, char jobz, char uplo, lapack_int n, float* A, lapack_int lda, float* B, lapack_int ldb, float* W )
{
    return LAPACKE_ssygvd( LAPACK_COL_MAJOR, itype, jobz, uplo, n, A, lda, B, ldb, W );
}

static lapack_int LAPACKE_hegvd(
    lapack_int itype, char jobz, char uplo, lapack_int n, double* A, lapack_int lda, double* B, lapack_int ldb, double* W )
{
    return LAPACKE_dsygvd( LAPACK_COL_MAJOR, itype, jobz, uplo, n, A, lda, B, ldb, W );
}

static lapack_int LAPACKE_hegvd(
    lapack_int itype, char jobz, char uplo, lapack_int n, std::complex<float>* A, lapack_int lda, std::complex<float>* B, lapack_int ldb, float* W )
{
    return LAPACKE_chegvd( LAPACK_COL_MAJOR, itype, jobz, uplo, n, A, lda, B, ldb, W );
}

static lapack_int LAPACKE_hegvd(
    lapack_int itype, char jobz, char uplo, lapack_int n, std::complex<double>* A, lapack_int lda, std::complex<double>* B, lapack_int ldb, double* W )
{
    return LAPACKE_zhegvd( LAPACK_COL_MAJOR, itype, jobz, uplo, n, A, lda, B, ldb, W );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_hegvd_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    int64_t itype = params.itype.value();
    lapack::Job jobz = params.jobz.value();
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
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
    size_t size_B = (size_t) ldb * n;
    size_t size_W = (size_t) (n);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );
    std::vector< real_t > W_tst( size_W );
    std::vector< real_t > W_ref( size_W );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );

    // diagonally dominant -> positive definite
    for (int64_t i = 0; i < n; ++i) {
        B_tst[ i + i*ldb ] += n;
    }

    A_ref = A_tst;
    B_ref = B_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::hegvd( itype, jobz, uplo, n, &A_tst[0], lda, &B_tst[0], ldb, &W_tst[0] );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::hegvd returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::hegvd( itype, jobz, n );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_hegvd( itype, job2char(jobz), uplo2char(uplo), n, &A_ref[0], lda, &B_ref[0], ldb, &W_ref[0] );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_hegvd returned error %lld\n", (lld) info_ref );
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
        error += abs_error( W_tst, W_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_hegvd( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_hegvd_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_hegvd_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_hegvd_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_hegvd_work< std::complex<double> >( params, run );
            break;
    }
}
