#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_poequ(
    lapack_int n, float* A, lapack_int lda, float* S, float* scond, float* amax )
{
    return LAPACKE_spoequ( LAPACK_COL_MAJOR, n, A, lda, S, scond, amax );
}

static lapack_int LAPACKE_poequ(
    lapack_int n, double* A, lapack_int lda, double* S, double* scond, double* amax )
{
    return LAPACKE_dpoequ( LAPACK_COL_MAJOR, n, A, lda, S, scond, amax );
}

static lapack_int LAPACKE_poequ(
    lapack_int n, std::complex<float>* A, lapack_int lda, float* S, float* scond, float* amax )
{
    return LAPACKE_cpoequ( LAPACK_COL_MAJOR, n, A, lda, S, scond, amax );
}

static lapack_int LAPACKE_poequ(
    lapack_int n, std::complex<double>* A, lapack_int lda, double* S, double* scond, double* amax )
{
    return LAPACKE_zpoequ( LAPACK_COL_MAJOR, n, A, lda, S, scond, amax );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_poequ_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    real_t scond_tst = 0;
    real_t scond_ref = 0;
    real_t amax_tst;
    real_t amax_ref;
    size_t size_A = (size_t) lda * n;
    size_t size_S = (size_t) (n);

    std::vector< scalar_t > A( size_A );
    std::vector< real_t > S_tst( size_S );
    std::vector< real_t > S_ref( size_S );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A.size(), &A[0] );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::poequ( n, &A[0], lda, &S_tst[0], &scond_tst, &amax_tst );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::poequ returned error %lld\n", (lld) info_tst );
    }

    //double gflop = lapack::Gflop< scalar_t >::poequ( n );
    params.time.value()   = time;
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_poequ( n, &A[0], lda, &S_ref[0], &scond_ref, &amax_ref );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_poequ returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value()   = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( S_tst, S_ref );
        error += std::abs( scond_tst - scond_ref );
        error += std::abs( amax_tst - amax_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_poequ( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_poequ_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_poequ_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_poequ_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_poequ_work< std::complex<double> >( params, run );
            break;
    }
}
