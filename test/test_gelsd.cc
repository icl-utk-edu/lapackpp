#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gelsd(
    lapack_int m, lapack_int n, lapack_int nrhs, float* A, lapack_int lda, float* B, lapack_int ldb, float* S, float rcond, lapack_int* rank )
{
    return LAPACKE_sgelsd( LAPACK_COL_MAJOR, m, n, nrhs, A, lda, B, ldb, S, rcond, rank );
}

static lapack_int LAPACKE_gelsd(
    lapack_int m, lapack_int n, lapack_int nrhs, double* A, lapack_int lda, double* B, lapack_int ldb, double* S, double rcond, lapack_int* rank )
{
    return LAPACKE_dgelsd( LAPACK_COL_MAJOR, m, n, nrhs, A, lda, B, ldb, S, rcond, rank );
}

static lapack_int LAPACKE_gelsd(
    lapack_int m, lapack_int n, lapack_int nrhs, std::complex<float>* A, lapack_int lda, std::complex<float>* B, lapack_int ldb, float* S, float rcond, lapack_int* rank )
{
    return LAPACKE_cgelsd( LAPACK_COL_MAJOR, m, n, nrhs, A, lda, B, ldb, S, rcond, rank );
}

static lapack_int LAPACKE_gelsd(
    lapack_int m, lapack_int n, lapack_int nrhs, std::complex<double>* A, lapack_int lda, std::complex<double>* B, lapack_int ldb, double* S, double rcond, lapack_int* rank )
{
    return LAPACKE_zgelsd( LAPACK_COL_MAJOR, m, n, nrhs, A, lda, B, ldb, S, rcond, rank );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gelsd_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs.value();
    int64_t align = params.align.value();
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, m ), align );
    int64_t ldb = roundup( max( max( 1, m ), n ), align );
    real_t rcond;
    int64_t rank_tst = 0;
    lapack_int rank_ref;
    size_t size_A = (size_t) ( lda *  n);
    size_t size_B = (size_t) ( ldb * nrhs );
    size_t size_S = (size_t) min( m, n );

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );
    std::vector< real_t > S_tst( size_S );
    std::vector< real_t > S_ref( size_S );

    lapack::generate_matrix( params.matrix, m, n, nullptr, &A_tst[0], lda );
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );

    A_ref = A_tst;
    B_ref = B_tst;

    // TODO: rcond value is set to a meaningless value, fix this
    rcond = n;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::gelsd( m, n, nrhs, &A_tst[0], lda, &B_tst[0], ldb, &S_tst[0], rcond, &rank_tst );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gelsd returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::gelsd( m, n, nrhs );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_gelsd( m, n, nrhs, &A_ref[0], lda, &B_ref[0], ldb, &S_ref[0], rcond, &rank_ref );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gelsd returned error %lld\n", (lld) info_ref );
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
        error += abs_error( S_tst, S_ref );
        error += std::abs( rank_tst - rank_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gelsd( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gelsd_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gelsd_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gelsd_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gelsd_work< std::complex<double> >( params, run );
            break;
    }
}
