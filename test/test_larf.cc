#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

extern "C" {

/* ----- apply Householder reflector */
#define LAPACK_slarf LAPACK_GLOBAL(slarf,SLARF)
void LAPACK_slarf(
    char const* side,
    lapack_int const* m, lapack_int const* n,
    float const* v, lapack_int const* incv,
    float const* tau,
    float* c, lapack_int const* ldc,
    float* work );
#define LAPACK_dlarf LAPACK_GLOBAL(dlarf,DLARF)
void LAPACK_dlarf(
    char const* side,
    lapack_int const* m, lapack_int const* n,
    double const* v, lapack_int const* incv,
    double const* tau,
    double* c, lapack_int const* ldc,
    double* work );
#define LAPACK_clarf LAPACK_GLOBAL(clarf,CLARF)
void LAPACK_clarf(
    char const* side,
    lapack_int const* m, lapack_int const* n,
    lapack_complex_float const* v, lapack_int const* incv,
    lapack_complex_float const* tau,
    lapack_complex_float* c, lapack_int const* ldc,
    lapack_complex_float* work );
#define LAPACK_zlarf LAPACK_GLOBAL(zlarf,ZLARF)
void LAPACK_zlarf(
    char const* side,
    lapack_int const* m, lapack_int const* n,
    lapack_complex_double const* v, lapack_int const* incv,
    lapack_complex_double const* tau,
    lapack_complex_double* c, lapack_int const* ldc,
    lapack_complex_double* work );

}  // extern "C"

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACK (not in LAPACKE)
// todo: LAPACK has no error checks for larf
static lapack_int LAPACKE_larf(
    char side, lapack_int m, lapack_int n, float* V, lapack_int incv, float tau, float* C, lapack_int ldc )
{
    std::vector< float > work( side == 'l' || side == 'L' ? n : m );
    LAPACK_slarf( &side, &m, &n, V, &incv, &tau, C, &ldc, &work[0] );
    return 0;
}

static lapack_int LAPACKE_larf(
    char side, lapack_int m, lapack_int n, double* V, lapack_int incv, double tau, double* C, lapack_int ldc )
{
    std::vector< double > work( side == 'l' || side == 'L' ? n : m );
    LAPACK_dlarf( &side, &m, &n, V, &incv, &tau, C, &ldc, &work[0] );
    return 0;
}

static lapack_int LAPACKE_larf(
    char side, lapack_int m, lapack_int n, std::complex<float>* V, lapack_int incv, std::complex<float> tau, std::complex<float>* C, lapack_int ldc )
{
    std::vector< std::complex<float> > work( side == 'l' || side == 'L' ? n : m );
    LAPACK_clarf( &side, &m, &n, V, &incv, &tau, C, &ldc, &work[0] );
    return 0;
}

static lapack_int LAPACKE_larf(
    char side, lapack_int m, lapack_int n, std::complex<double>* V, lapack_int incv, std::complex<double> tau, std::complex<double>* C, lapack_int ldc )
{
    std::vector< std::complex<double> > work( side == 'l' || side == 'L' ? n : m );
    LAPACK_zlarf( &side, &m, &n, V, &incv, &tau, C, &ldc, &work[0] );
    return 0;
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_larf_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Side side = params.side.value();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t incv = params.incx.value();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();

    if (! run)
        return;

    // ---------- setup
    scalar_t tau;
    int64_t ldc = roundup( max( 1, m ), align );
    size_t size_V;
    if (side == lapack::Side::Left)
        size_V = 1 + (m-1)*abs(incv);
    else
        size_V = 1 + (n-1)*abs(incv);
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
    double time = omp_get_wtime();
    lapack::larf( side, m, n, &V[0], incv, tau, &C_tst[0], ldc );
    time = omp_get_wtime() - time;

    //double gflop = lapack::Gflop< scalar_t >::larf( side, m, n );
    params.time.value()   = time;
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_larf( side2char(side), m, n, &V[0], incv, tau, &C_ref[0], ldc );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_larf returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value()   = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        error += abs_error( C_tst, C_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_larf( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_larf_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_larf_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_larf_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_larf_work< std::complex<double> >( params, run );
            break;
    }
}
