#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gbtrf(
    lapack_int m, lapack_int n, lapack_int kl, lapack_int ku, float* AB, lapack_int ldab, lapack_int* ipiv )
{
    return LAPACKE_sgbtrf( LAPACK_COL_MAJOR, m, n, kl, ku, AB, ldab, ipiv );
}

static lapack_int LAPACKE_gbtrf(
    lapack_int m, lapack_int n, lapack_int kl, lapack_int ku, double* AB, lapack_int ldab, lapack_int* ipiv )
{
    return LAPACKE_dgbtrf( LAPACK_COL_MAJOR, m, n, kl, ku, AB, ldab, ipiv );
}

static lapack_int LAPACKE_gbtrf(
    lapack_int m, lapack_int n, lapack_int kl, lapack_int ku, std::complex<float>* AB, lapack_int ldab, lapack_int* ipiv )
{
    return LAPACKE_cgbtrf( LAPACK_COL_MAJOR, m, n, kl, ku, AB, ldab, ipiv );
}

static lapack_int LAPACKE_gbtrf(
    lapack_int m, lapack_int n, lapack_int kl, lapack_int ku, std::complex<double>* AB, lapack_int ldab, lapack_int* ipiv )
{
    return LAPACKE_zgbtrf( LAPACK_COL_MAJOR, m, n, kl, ku, AB, ldab, ipiv );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gbtrf_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
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
    int64_t ldab = roundup( 2*kl+ku+1, align );
    size_t size_AB = (size_t) ldab * n;
    size_t size_ipiv = (size_t) (min(m,n));

    std::vector< scalar_t > AB_tst( size_AB );
    std::vector< scalar_t > AB_ref( size_AB );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB_tst.size(), &AB_tst[0] );
    AB_ref = AB_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::gbtrf( m, n, kl, ku, &AB_tst[0], ldab, &ipiv_tst[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gbtrf returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::gbtrf( m, n, kl, ku );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_gbtrf( m, n, kl, ku, &AB_ref[0], ldab, &ipiv_ref[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gbtrf returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( AB_tst, AB_ref );
        error += abs_error( ipiv_tst, ipiv_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gbtrf( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gbtrf_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gbtrf_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gbtrf_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gbtrf_work< std::complex<double> >( params, run );
            break;
    }
}
