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
static lapack_int LAPACKE_getri(
    lapack_int n, float* A, lapack_int lda, lapack_int* ipiv )
{
    return LAPACKE_sgetri( LAPACK_COL_MAJOR, n, A, lda, ipiv );
}

static lapack_int LAPACKE_getri(
    lapack_int n, double* A, lapack_int lda, lapack_int* ipiv )
{
    return LAPACKE_dgetri( LAPACK_COL_MAJOR, n, A, lda, ipiv );
}

static lapack_int LAPACKE_getri(
    lapack_int n, std::complex<float>* A, lapack_int lda, lapack_int* ipiv )
{
    return LAPACKE_cgetri( LAPACK_COL_MAJOR, n, A, lda, ipiv );
}

static lapack_int LAPACKE_getri(
    lapack_int n, std::complex<double>* A, lapack_int lda, lapack_int* ipiv )
{
    return LAPACKE_zgetri( LAPACK_COL_MAJOR, n, A, lda, ipiv );
}

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_getrf(
    lapack_int m, lapack_int n, float* A, lapack_int lda, lapack_int* ipiv )
{
    return LAPACKE_sgetrf( LAPACK_COL_MAJOR, m, n, A, lda, ipiv );
}

static lapack_int LAPACKE_getrf(
    lapack_int m, lapack_int n, double* A, lapack_int lda, lapack_int* ipiv )
{
    return LAPACKE_dgetrf( LAPACK_COL_MAJOR, m, n, A, lda, ipiv );
}

static lapack_int LAPACKE_getrf(
    lapack_int m, lapack_int n, std::complex<float>* A, lapack_int lda, lapack_int* ipiv )
{
    return LAPACKE_cgetrf( LAPACK_COL_MAJOR, m, n, A, lda, ipiv );
}

static lapack_int LAPACKE_getrf(
    lapack_int m, lapack_int n, std::complex<double>* A, lapack_int lda, lapack_int* ipiv )
{
    return LAPACKE_zgetrf( LAPACK_COL_MAJOR, m, n, A, lda, ipiv );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_getri_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_ipiv = (size_t) (n);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    A_ref = A_tst;

    // factor A into LU
    int64_t info = lapack::getrf( n, n, &A_tst[0], lda, &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::getrf returned error %lld\n", (lld) info );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::getri( n, &A_tst[0], lda, &ipiv_tst[0] );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::getri returned error %lld\n", (lld) info_tst );
    }

    double gflop = lapack::Gflop< scalar_t >::getri( n );
    params.time.value()   = time;
    params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // factor A into LU
        info = LAPACKE_getrf( n, n, &A_ref[0], lda, &ipiv_ref[0] );
        if (info != 0) {
            fprintf( stderr, "LAPACKE_getrf returned error %lld\n", (lld) info );
        }

        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_getri( n, &A_ref[0], lda, &ipiv_ref[0] );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_getri returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value()   = time;
        params.ref_gflops.value() = gflop / time;

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
void test_getri( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_getri_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_getri_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_getri_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_getri_work< std::complex<double> >( params, run );
            break;
    }
}
