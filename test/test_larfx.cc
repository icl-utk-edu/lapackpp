#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_larfx(
    char side, lapack_int m, lapack_int n, float* V, float tau, float* C, lapack_int ldc )
{
    std::vector< float > work( side == 'l' || side == 'L' ? n : m );
    return LAPACKE_slarfx( LAPACK_COL_MAJOR, side, m, n, V, tau, C, ldc, &work[0] );
}

static lapack_int LAPACKE_larfx(
    char side, lapack_int m, lapack_int n, double* V, double tau, double* C, lapack_int ldc )
{
    std::vector< double > work( side == 'l' || side == 'L' ? n : m );
    return LAPACKE_dlarfx( LAPACK_COL_MAJOR, side, m, n, V, tau, C, ldc, &work[0] );
}

static lapack_int LAPACKE_larfx(
    char side, lapack_int m, lapack_int n, std::complex<float>* V, std::complex<float> tau, std::complex<float>* C, lapack_int ldc )
{
    std::vector< std::complex<float> > work( side == 'l' || side == 'L' ? n : m );
    return LAPACKE_clarfx( LAPACK_COL_MAJOR, side, m, n, V, tau, C, ldc, &work[0] );
}

static lapack_int LAPACKE_larfx(
    char side, lapack_int m, lapack_int n, std::complex<double>* V, std::complex<double> tau, std::complex<double>* C, lapack_int ldc )
{
    std::vector< std::complex<double> > work( side == 'l' || side == 'L' ? n : m );
    return LAPACKE_zlarfx( LAPACK_COL_MAJOR, side, m, n, V, tau, C, ldc, &work[0] );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_larfx_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Side side = params.side.value();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    scalar_t tau;
    int64_t ldc = roundup( max( 1, m ), align );
    size_t size_V;
    if (side == lapack::Side::Left)
        size_V = m;
    else
        size_V = n;
    size_t size_C = (size_t) ldc * n;

    std::vector< scalar_t > V( size_V );
    std::vector< scalar_t > C_tst( size_C );
    std::vector< scalar_t > C_ref( size_C );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, V.size(), &V[0] );
    lapack::larnv( idist, iseed, 1, &tau );
    lapack::larnv( idist, iseed, C_tst.size(), &C_tst[0] );
    C_ref = C_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    lapack::larfx( side, m, n, &V[0], tau, &C_tst[0], ldc );
    time = get_wtime() - time;

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::larf( side, m, n );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_larfx( side2char(side), m, n, &V[0], tau, &C_ref[0], ldc );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_larfx returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        error += abs_error( C_tst, C_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_larfx( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_larfx_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_larfx_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_larfx_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_larfx_work< std::complex<double> >( params, run );
            break;
    }
}
