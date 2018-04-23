#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gels(
    char trans, lapack_int m, lapack_int n, lapack_int nrhs, float* A, lapack_int lda, float* B, lapack_int ldb )
{
    if ( trans == 'C' ) trans = 'T';
    return LAPACKE_sgels( LAPACK_COL_MAJOR, trans, m, n, nrhs, A, lda, B, ldb );
}

static lapack_int LAPACKE_gels(
    char trans, lapack_int m, lapack_int n, lapack_int nrhs, double* A, lapack_int lda, double* B, lapack_int ldb )
{
    if ( trans == 'C' ) trans = 'T';
    return LAPACKE_dgels( LAPACK_COL_MAJOR, trans, m, n, nrhs, A, lda, B, ldb );
}

static lapack_int LAPACKE_gels(
    char trans, lapack_int m, lapack_int n, lapack_int nrhs, std::complex<float>* A, lapack_int lda, std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cgels( LAPACK_COL_MAJOR, trans, m, n, nrhs, A, lda, B, ldb );
}

static lapack_int LAPACKE_gels(
    char trans, lapack_int m, lapack_int n, lapack_int nrhs, std::complex<double>* A, lapack_int lda, std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zgels( LAPACK_COL_MAJOR, trans, m, n, nrhs, A, lda, B, ldb );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gels_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Op trans = params.trans.value();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs.value();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    params.matrix.name.value();
    params.matrix.cond.value();
    params.matrix.condD.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, m ), align );
    int64_t ldb = roundup( max( max( 1, m ), n), align );
    size_t size_A = (size_t) lda * n;
    size_t size_B = (size_t) ldb * nrhs;

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    lapack::generate_matrix( params.matrix, m, n, nullptr, &A_tst[0], lda );
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );

    A_ref = A_tst;
    B_ref = B_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::gels( trans, m, n, nrhs, &A_tst[0], lda, &B_tst[0], ldb );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gels returned error %lld\n", (lld) info_tst );
    }

    // double gflop = lapack::Gflop< scalar_t >::gels( trans, m, n, nrhs );
    params.time.value()   = time;
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_gels( op2char(trans), m, n, nrhs, &A_ref[0], lda, &B_ref[0], ldb );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gels returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value()   = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        real_t eps = std::numeric_limits< real_t >::epsilon();
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( B_tst, B_ref );
        params.error.value() = error;
        params.okay.value() = (error < 3*eps);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gels( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gels_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gels_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gels_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gels_work< std::complex<double> >( params, run );
            break;
    }
}
