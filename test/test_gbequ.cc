#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gbequ(
    lapack_int m, lapack_int n, lapack_int kl, lapack_int ku, float* AB, lapack_int ldab, float* R, float* C, float* rowcnd, float* colcnd, float* amax )
{
    return LAPACKE_sgbequ( LAPACK_COL_MAJOR, m, n, kl, ku, AB, ldab, R, C, rowcnd, colcnd, amax );
}

static lapack_int LAPACKE_gbequ(
    lapack_int m, lapack_int n, lapack_int kl, lapack_int ku, double* AB, lapack_int ldab, double* R, double* C, double* rowcnd, double* colcnd, double* amax )
{
    return LAPACKE_dgbequ( LAPACK_COL_MAJOR, m, n, kl, ku, AB, ldab, R, C, rowcnd, colcnd, amax );
}

static lapack_int LAPACKE_gbequ(
    lapack_int m, lapack_int n, lapack_int kl, lapack_int ku, std::complex<float>* AB, lapack_int ldab, float* R, float* C, float* rowcnd, float* colcnd, float* amax )
{
    return LAPACKE_cgbequ( LAPACK_COL_MAJOR, m, n, kl, ku, AB, ldab, R, C, rowcnd, colcnd, amax );
}

static lapack_int LAPACKE_gbequ(
    lapack_int m, lapack_int n, lapack_int kl, lapack_int ku, std::complex<double>* AB, lapack_int ldab, double* R, double* C, double* rowcnd, double* colcnd, double* amax )
{
    return LAPACKE_zgbequ( LAPACK_COL_MAJOR, m, n, kl, ku, AB, ldab, R, C, rowcnd, colcnd, amax );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gbequ_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t kl = params.kl.value();
    int64_t ku = params.ku.value();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( kl+ku+1, align );
    real_t rowcnd_tst = 0;
    real_t rowcnd_ref = 0;
    real_t colcnd_tst = 0;
    real_t colcnd_ref = 0;
    real_t amax_tst;
    real_t amax_ref;
    size_t size_AB = (size_t) ldab * n;
    size_t size_R = (size_t) (m);
    size_t size_C = (size_t) (n);

    std::vector< scalar_t > AB( size_AB );
    std::vector< real_t > R_tst( size_R );
    std::vector< real_t > R_ref( size_R );
    std::vector< real_t > C_tst( size_C );
    std::vector< real_t > C_ref( size_C );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB.size(), &AB[0] );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::gbequ( m, n, kl, ku, &AB[0], ldab, &R_tst[0], &C_tst[0], &rowcnd_tst, &colcnd_tst, &amax_tst );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gbequ returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::gbequ( m, n, kl, ku );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_gbequ( m, n, kl, ku, &AB[0], ldab, &R_ref[0], &C_ref[0], &rowcnd_ref, &colcnd_ref, &amax_ref );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gbequ returned error %lld\n", (lld) info_ref );
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
void test_gbequ( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gbequ_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gbequ_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gbequ_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gbequ_work< std::complex<double> >( params, run );
            break;
    }
}
