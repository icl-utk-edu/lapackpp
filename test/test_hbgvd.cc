#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_hbgvd(
    char jobz, char uplo, lapack_int n, lapack_int ka, lapack_int kb, float* AB, lapack_int ldab, float* BB, lapack_int ldbb, float* W, float* Z, lapack_int ldz )
{
    return LAPACKE_ssbgvd( LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb, AB, ldab, BB, ldbb, W, Z, ldz );
}

static lapack_int LAPACKE_hbgvd(
    char jobz, char uplo, lapack_int n, lapack_int ka, lapack_int kb, double* AB, lapack_int ldab, double* BB, lapack_int ldbb, double* W, double* Z, lapack_int ldz )
{
    return LAPACKE_dsbgvd( LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb, AB, ldab, BB, ldbb, W, Z, ldz );
}

static lapack_int LAPACKE_hbgvd(
    char jobz, char uplo, lapack_int n, lapack_int ka, lapack_int kb, std::complex<float>* AB, lapack_int ldab, std::complex<float>* BB, lapack_int ldbb, float* W, std::complex<float>* Z, lapack_int ldz )
{
    return LAPACKE_chbgvd( LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb, AB, ldab, BB, ldbb, W, Z, ldz );
}
static lapack_int LAPACKE_hbgvd(
    char jobz, char uplo, lapack_int n, lapack_int ka, lapack_int kb, std::complex<double>* AB, lapack_int ldab, std::complex<double>* BB, lapack_int ldbb, double* W, std::complex<double>* Z, lapack_int ldz )
{
    return LAPACKE_zhbgvd( LAPACK_COL_MAJOR, jobz, uplo, n, ka, kb, AB, ldab, BB, ldbb, W, Z, ldz );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_hbgvd_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Job jobz = params.jobz.value();
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    int64_t ka = params.kd.value();
    int64_t kb = params.kd.value();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( ka+1, align );
    int64_t ldbb = roundup( kb+1, align );
    int64_t ldz = ( jobz==lapack::Job::Vec ? roundup( max( 1, n ), align ) : align );
    size_t size_AB = (size_t) (ldab * n);
    size_t size_BB = (size_t) (ldbb * n);
    size_t size_W = (size_t) (n);
    size_t size_Z = (size_t) (ldz * n);

    std::vector< scalar_t > AB_tst( size_AB );
    std::vector< scalar_t > AB_ref( size_AB );
    std::vector< scalar_t > BB_tst( size_BB );
    std::vector< scalar_t > BB_ref( size_BB );
    std::vector< real_t > W_tst( size_W );
    std::vector< real_t > W_ref( size_W );
    std::vector< scalar_t > Z_tst( size_Z );
    std::vector< scalar_t > Z_ref( size_Z );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB_tst.size(), &AB_tst[0] );
    lapack::larnv( idist, iseed, BB_tst.size(), &BB_tst[0] );

    // diagonally dominant -> positive definite
    if (uplo == lapack::Uplo::Upper) {
        for (int64_t j = 0; j < n; ++j) {
            BB_tst[ kb + j*ldbb ] += n;
        }
    } else { // lower
       for (int64_t j = 0; j < n; ++j) {
           BB_tst[ j*ldbb ] += n;
       }
    }

    AB_ref = AB_tst;
    BB_ref = BB_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::hbgvd( jobz, uplo, n, ka, kb, &AB_tst[0], ldab, &BB_tst[0], ldbb, &W_tst[0], &Z_tst[0], ldz );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::hbgvd returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::hbgvd( jobz, n, ka, kb );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_hbgvd( job2char(jobz), uplo2char(uplo), n, ka, kb, &AB_ref[0], ldab, &BB_ref[0], ldbb, &W_ref[0], &Z_ref[0], ldz );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_hbgvd returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( AB_tst, AB_ref );
        error += abs_error( BB_tst, BB_ref );
        error += abs_error( W_tst, W_ref );
        error += abs_error( Z_tst, Z_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_hbgvd( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_hbgvd_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_hbgvd_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_hbgvd_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_hbgvd_work< std::complex<double> >( params, run );
            break;
    }
}
