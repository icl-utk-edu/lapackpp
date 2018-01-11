#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_pbsv(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs, float* AB, lapack_int ldab, float* B, lapack_int ldb )
{
    return LAPACKE_spbsv( LAPACK_COL_MAJOR, uplo, n, kd, nrhs, AB, ldab, B, ldb );
}

static lapack_int LAPACKE_pbsv(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs, double* AB, lapack_int ldab, double* B, lapack_int ldb )
{
    return LAPACKE_dpbsv( LAPACK_COL_MAJOR, uplo, n, kd, nrhs, AB, ldab, B, ldb );
}

static lapack_int LAPACKE_pbsv(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs, std::complex<float>* AB, lapack_int ldab, std::complex<float>* B, lapack_int ldb )
{
    return LAPACKE_cpbsv( LAPACK_COL_MAJOR, uplo, n, kd, nrhs, AB, ldab, B, ldb );
}

static lapack_int LAPACKE_pbsv(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs, std::complex<double>* AB, lapack_int ldab, std::complex<double>* B, lapack_int ldb )
{
    return LAPACKE_zpbsv( LAPACK_COL_MAJOR, uplo, n, kd, nrhs, AB, ldab, B, ldb );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_pbsv_work( Params& params, bool run )
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

    std::vector< scalar_t > AB_tst( size_AB );
    std::vector< scalar_t > AB_ref( size_AB );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB_tst.size(), &AB_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );

    // diagonally dominant -> positive definite
    if (uplo == lapack::Uplo::Upper) {
        for (int64_t j = 0; j < n; ++j) {
            AB_tst[ kd + j*ldab ] += n;
        }
    }
    else { // lower
       for (int64_t j = 0; j < n; ++j) {
           AB_tst[ j*ldab ] += n;
       }
    }

    AB_ref = AB_tst;
    B_ref = B_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::pbsv( uplo, n, kd, nrhs, &AB_tst[0], ldab, &B_tst[0], ldb );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::pbsv returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::pbsv( n, kd, nrhs );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_pbsv( uplo2char(uplo), n, kd, nrhs, &AB_ref[0], ldab, &B_ref[0], ldb );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_pbsv returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( AB_tst, AB_ref );
        error += abs_error( B_tst, B_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_pbsv( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_pbsv_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_pbsv_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_pbsv_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_pbsv_work< std::complex<double> >( params, run );
            break;
    }
}
