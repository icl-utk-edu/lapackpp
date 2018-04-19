#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_unmtr(
    char side, char uplo, char trans, lapack_int m, lapack_int n, float* A, lapack_int lda, float* tau, float* C, lapack_int ldc )
{
    if (trans == 'C') trans = 'T';
    return LAPACKE_sormtr( LAPACK_COL_MAJOR, side, uplo, trans, m, n, A, lda, tau, C, ldc );
}

static lapack_int LAPACKE_unmtr(
    char side, char uplo, char trans, lapack_int m, lapack_int n, double* A, lapack_int lda, double* tau, double* C, lapack_int ldc )
{
    if (trans == 'C') trans = 'T';
    return LAPACKE_dormtr( LAPACK_COL_MAJOR, side, uplo, trans, m, n, A, lda, tau, C, ldc );
}

static lapack_int LAPACKE_unmtr(
    char side, char uplo, char trans, lapack_int m, lapack_int n, std::complex<float>* A, lapack_int lda, std::complex<float>* tau, std::complex<float>* C, lapack_int ldc )
{
    return LAPACKE_cunmtr( LAPACK_COL_MAJOR, side, uplo, trans, m, n, A, lda, tau, C, ldc );
}

static lapack_int LAPACKE_unmtr(
    char side, char uplo, char trans, lapack_int m, lapack_int n, std::complex<double>* A, lapack_int lda, std::complex<double>* tau, std::complex<double>* C, lapack_int ldc )
{
    return LAPACKE_zunmtr( LAPACK_COL_MAJOR, side, uplo, trans, m, n, A, lda, tau, C, ldc );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_unmtr_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Side side = params.side.value();
    lapack::Uplo uplo = params.uplo.value();
    lapack::Op trans = params.trans.value();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t r = (side == lapack::Side::Left) ? m : n;
    int64_t lda = roundup( max( 1, r ), align );
    int64_t ldc = roundup( max( 1, m ), align );
    size_t size_A = (size_t) ( max( 1, lda*r ) );
    size_t size_tau = (size_t) ( max( 1, r-1 ) );
    size_t size_C = (size_t) max( 1, ldc * n );
    size_t size_D = (size_t) (r);
    size_t size_E = (size_t) (r-1);

    std::vector< scalar_t > A( size_A );
    std::vector< scalar_t > tau( size_tau );
    std::vector< scalar_t > C_tst( size_C );
    std::vector< scalar_t > C_ref( size_C );
    std::vector< real_t > D( size_D );
    std::vector< real_t > E( size_E );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A.size(), &A[0] );
    lapack::larnv( idist, iseed, tau.size(), &tau[0] );
    lapack::larnv( idist, iseed, C_tst.size(), &C_tst[0] );
    C_ref = C_tst;

    int64_t info = lapack::hetrd( uplo, r, &A[0], lda, &D[0], &E[0], &tau[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::hetrd returned error %lld\n", (lld) info );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::unmtr( side, uplo, trans, m, n, &A[0], lda, &tau[0], &C_tst[0], ldc );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::unmtr returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::unmtr( side, trans, m, n );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_unmtr( side2char(side), uplo2char(uplo), op2char(trans), m, n, &A[0], lda, &tau[0], &C_ref[0], ldc );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_unmtr returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( C_tst, C_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_unmtr( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_unmtr_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_unmtr_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_unmtr_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_unmtr_work< std::complex<double> >( params, run );
            break;
    }
}
