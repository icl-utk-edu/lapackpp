#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_pbrfs(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs, float* AB, lapack_int ldab, float* AFB, lapack_int ldafb, float* B, lapack_int ldb, float* X, lapack_int ldx, float* ferr, float* berr )
{
    return LAPACKE_spbrfs( LAPACK_COL_MAJOR, uplo, n, kd, nrhs, AB, ldab, AFB, ldafb, B, ldb, X, ldx, ferr, berr );
}

static lapack_int LAPACKE_pbrfs(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs, double* AB, lapack_int ldab, double* AFB, lapack_int ldafb, double* B, lapack_int ldb, double* X, lapack_int ldx, double* ferr, double* berr )
{
    return LAPACKE_dpbrfs( LAPACK_COL_MAJOR, uplo, n, kd, nrhs, AB, ldab, AFB, ldafb, B, ldb, X, ldx, ferr, berr );
}

static lapack_int LAPACKE_pbrfs(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs, std::complex<float>* AB, lapack_int ldab, std::complex<float>* AFB, lapack_int ldafb, std::complex<float>* B, lapack_int ldb, std::complex<float>* X, lapack_int ldx, float* ferr, float* berr )
{
    return LAPACKE_cpbrfs( LAPACK_COL_MAJOR, uplo, n, kd, nrhs, AB, ldab, AFB, ldafb, B, ldb, X, ldx, ferr, berr );
}

static lapack_int LAPACKE_pbrfs(
    char uplo, lapack_int n, lapack_int kd, lapack_int nrhs, std::complex<double>* AB, lapack_int ldab, std::complex<double>* AFB, lapack_int ldafb, std::complex<double>* B, lapack_int ldb, std::complex<double>* X, lapack_int ldx, double* ferr, double* berr )
{
    return LAPACKE_zpbrfs( LAPACK_COL_MAJOR, uplo, n, kd, nrhs, AB, ldab, AFB, ldafb, B, ldb, X, ldx, ferr, berr );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_pbrfs_work( Params& params, bool run )
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

    real_t eps = std::numeric_limits< real_t >::epsilon();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( kd+1, align );
    int64_t ldafb = roundup( kd+1, align );
    int64_t ldb = roundup( max( 1, n ), align );
    int64_t ldx = roundup( max( 1, n ), align );
    size_t size_AB = (size_t) ldab * n;
    size_t size_AFB = (size_t) ldafb * n;
    size_t size_B = (size_t) ldb * nrhs;
    size_t size_X = (size_t) ldx * nrhs;
    size_t size_ferr = (size_t) (nrhs);
    size_t size_berr = (size_t) (nrhs);

    std::vector< scalar_t > AB( size_AB );
    std::vector< scalar_t > AFB( size_AFB );
    std::vector< scalar_t > B( size_B );
    std::vector< scalar_t > X_tst( size_X );
    std::vector< scalar_t > X_ref( size_X );
    std::vector< real_t > ferr_tst( size_ferr );
    std::vector< real_t > ferr_ref( size_ferr );
    std::vector< real_t > berr_tst( size_berr );
    std::vector< real_t > berr_ref( size_berr );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB.size(), &AB[0] );
    lapack::larnv( idist, iseed, AFB.size(), &AFB[0] );
    lapack::larnv( idist, iseed, B.size(), &B[0] );
    lapack::larnv( idist, iseed, X_tst.size(), &X_tst[0] );
    X_ref = X_tst;

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

    AFB = AB;

    // factor
    int64_t info = lapack::pbtrf( uplo, n, kd, &AFB[0], ldab );
    if (info != 0) {
        fprintf( stderr, "lapack::pbtrf returned error %lld\n", (lld) info );
    }

    // compute solution
    info = lapack::pbtrs ( uplo, n, kd, nrhs, &AFB[0], ldab, &B[0], ldb );
    if (info != 0) {
        fprintf( stderr, "lapack::pbtrs returned error %lld\n", (lld) info );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::pbrfs( uplo, n, kd, nrhs, &AB[0], ldab, &AFB[0], ldafb, &B[0], ldb, &X_tst[0], ldx, &ferr_tst[0], &berr_tst[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::pbrfs returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::pbrfs( n, kd, nrhs );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_pbrfs( uplo2char(uplo), n, kd, nrhs, &AB[0], ldab, &AFB[0], ldafb, &B[0], ldb, &X_ref[0], ldx, &ferr_ref[0], &berr_ref[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_pbrfs returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( X_tst, X_ref );
        error += abs_error( ferr_tst, ferr_ref );
        error += abs_error( berr_tst, berr_ref );
        params.error.value() = error;
        params.okay.value() = (error < 3*eps);
    }
}

// -----------------------------------------------------------------------------
void test_pbrfs( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_pbrfs_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_pbrfs_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_pbrfs_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_pbrfs_work< std::complex<double> >( params, run );
            break;
    }
}
