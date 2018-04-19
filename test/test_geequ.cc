#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_geequ(
    lapack_int m, lapack_int n, float* A, lapack_int lda, float* R, float* C, float* rowcnd, float* colcnd, float* amax )
{
    return LAPACKE_sgeequ( LAPACK_COL_MAJOR, m, n, A, lda, R, C, rowcnd, colcnd, amax );
}

static lapack_int LAPACKE_geequ(
    lapack_int m, lapack_int n, double* A, lapack_int lda, double* R, double* C, double* rowcnd, double* colcnd, double* amax )
{
    return LAPACKE_dgeequ( LAPACK_COL_MAJOR, m, n, A, lda, R, C, rowcnd, colcnd, amax );
}

static lapack_int LAPACKE_geequ(
    lapack_int m, lapack_int n, std::complex<float>* A, lapack_int lda, float* R, float* C, float* rowcnd, float* colcnd, float* amax )
{
    return LAPACKE_cgeequ( LAPACK_COL_MAJOR, m, n, A, lda, R, C, rowcnd, colcnd, amax );
}

static lapack_int LAPACKE_geequ(
    lapack_int m, lapack_int n, std::complex<double>* A, lapack_int lda, double* R, double* C, double* rowcnd, double* colcnd, double* amax )
{
    return LAPACKE_zgeequ( LAPACK_COL_MAJOR, m, n, A, lda, R, C, rowcnd, colcnd, amax );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_geequ_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();
    params.matrix.name.value();
    params.matrix.cond.value();
    params.matrix.condD.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, m ), align );
    real_t rowcnd_tst = 0;
    real_t rowcnd_ref = 0;
    real_t colcnd_tst = 0;
    real_t colcnd_ref = 0;
    real_t amax_tst;
    real_t amax_ref;
    size_t size_A = (size_t) lda * n;
    size_t size_R = (size_t) (m);
    size_t size_C = (size_t) (n);

    std::vector< scalar_t > A( size_A );
    std::vector< real_t > R_tst( size_R );
    std::vector< real_t > R_ref( size_R );
    std::vector< real_t > C_tst( size_C );
    std::vector< real_t > C_ref( size_C );

    lapack_generate_matrix( params.matrix, m, n, nullptr, &A[0], lda );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::geequ( m, n, &A[0], lda, &R_tst[0], &C_tst[0], &rowcnd_tst, &colcnd_tst, &amax_tst );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::geequ returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::geequ( m, n );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_geequ( m, n, &A[0], lda, &R_ref[0], &C_ref[0], &rowcnd_ref, &colcnd_ref, &amax_ref );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_geequ returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( R_tst, R_ref );
        error += abs_error( C_tst, C_ref );
        error += std::abs( rowcnd_tst - rowcnd_ref );
        error += std::abs( colcnd_tst - colcnd_ref );
        error += std::abs( amax_tst - amax_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_geequ( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_geequ_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_geequ_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_geequ_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_geequ_work< std::complex<double> >( params, run );
            break;
    }
}
