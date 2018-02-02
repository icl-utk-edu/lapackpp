#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gbsv(
    lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs, float* AB, lapack_int ldab, lapack_int* ipiv, float* B, lapack_int ldb )
{
    return LAPACKE_sgbsv( LAPACK_COL_MAJOR, n, kl, ku, nrhs, AB, ldab, ipiv, B, ldb );
}

static lapack_int LAPACKE_gbsv(
    lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs, double* AB, lapack_int ldab, lapack_int* ipiv, double* B, lapack_int ldb )
{
    return LAPACKE_dgbsv( LAPACK_COL_MAJOR, n, kl, ku, nrhs, AB, ldab, ipiv, B, ldb );
}

static lapack_int LAPACKE_gbsv(
    lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs, std::complex<float>* AB, lapack_int ldab, lapack_int* ipiv, std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cgbsv( LAPACK_COL_MAJOR, n, kl, ku, nrhs, AB, ldab, ipiv, B, ldb );
}

static lapack_int LAPACKE_gbsv(
    lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs, std::complex<double>* AB, lapack_int ldab, lapack_int* ipiv, std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zgbsv( LAPACK_COL_MAJOR, n, kl, ku, nrhs, AB, ldab, ipiv, B, ldb );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gbsv_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t kl = params.kl.value();
    int64_t ku = params.ku.value();
    int64_t nrhs = params.nrhs.value();
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
    int64_t ldab = roundup( 2*kl+ku+1, align );
    int64_t ldb = roundup( max( 1, n ), align );
    size_t size_AB = (size_t) ldab * n;
    size_t size_ipiv = (size_t) (n);
    size_t size_B = (size_t) ldb * nrhs;

    std::vector< scalar_t > AB_tst( size_AB );
    std::vector< scalar_t > AB_ref( size_AB );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    lapack_generate_matrix( params.matrix, ldab, n, nullptr, &AB_tst[0], ldab );
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    AB_ref = AB_tst;
    B_ref = B_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::gbsv( n, kl, ku, nrhs, &AB_tst[0], ldab, &ipiv_tst[0], &B_tst[0], ldb );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gbsv returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::gbsv( n, kl, ku, nrhs );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_gbsv( n, kl, ku, nrhs, &AB_ref[0], ldab, &ipiv_ref[0], &B_ref[0], ldb );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gbsv returned error %lld\n", (lld) info_ref );
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
        error += abs_error( B_tst, B_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gbsv( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gbsv_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gbsv_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gbsv_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gbsv_work< std::complex<double> >( params, run );
            break;
    }
}
