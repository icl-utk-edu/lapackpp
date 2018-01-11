#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_pbtrs(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs, float* AB, lapack_int ldab, float* B, lapack_int ldb )
{
    return LAPACKE_spbtrs( LAPACK_COL_MAJOR, uplo, n, kd, nrhs, AB, ldab, B, ldb );
}

static lapack_int LAPACKE_pbtrs(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs, double* AB, lapack_int ldab, double* B, lapack_int ldb )
{
    return LAPACKE_dpbtrs( LAPACK_COL_MAJOR, uplo, n, kd, nrhs, AB, ldab, B, ldb );
}

static lapack_int LAPACKE_pbtrs(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs, std::complex<float>* AB, lapack_int ldab, std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cpbtrs( LAPACK_COL_MAJOR, uplo, n, kd, nrhs, AB, ldab, B, ldb );
}

static lapack_int LAPACKE_pbtrs(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs, std::complex<double>* AB, lapack_int ldab, std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zpbtrs( LAPACK_COL_MAJOR, uplo, n, kd, nrhs, AB, ldab, B, ldb );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_pbtrs_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    int64_t kd = params.kd.value();
    int64_t nrhs = params.nrhs.value();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( kd+1, align );
    int64_t ldb = roundup( max( 1, n ), align );
    size_t size_AB = (size_t) ldab * n;
    size_t size_B = (size_t) ldb * nrhs;

    std::vector< scalar_t > AB( size_AB );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB.size(), &AB[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );

    // diagonally dominant -> positive definite
    if (uplo == lapack::Uplo::Upper) {
        for (int64_t j = 0; j < n; ++j) {
            AB[ kd + j*ldab ] += n;
        }
    }
    else { // lower
        for (int64_t j = 0; j < n; ++j) {
            AB[ j*ldab ] += n;
        }
    }

    B_ref = B_tst;

    // factor AB
    int64_t info = lapack::pbtrf( uplo, n, kd, &AB[0], ldab );
    if (info != 0) {
        fprintf( stderr, "lapack::pbtrf returned error %lldpn", (lld) info );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::pbtrs( uplo, n, kd, nrhs, &AB[0], ldab, &B_tst[0], ldb );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::pbtrs returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::pbtrs( n, kd, nrhs );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_pbtrs( uplo2char(uplo), n, kd, nrhs, &AB[0], ldab, &B_ref[0], ldb );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_pbtrs returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( B_tst, B_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_pbtrs( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_pbtrs_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_pbtrs_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_pbtrs_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_pbtrs_work< std::complex<double> >( params, run );
            break;
    }
}
