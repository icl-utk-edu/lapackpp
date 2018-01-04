#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gbrfs(
    char trans, lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs, float* AB, lapack_int ldab, float* AFB, lapack_int ldafb, lapack_int* ipiv, float* B, lapack_int ldb, float* X, lapack_int ldx, float* ferr, float* berr )
{
    return LAPACKE_sgbrfs( LAPACK_COL_MAJOR, trans, n, kl, ku, nrhs, AB, ldab, AFB, ldafb, ipiv, B, ldb, X, ldx, ferr, berr );
}

static lapack_int LAPACKE_gbrfs(
    char trans, lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs, double* AB, lapack_int ldab, double* AFB, lapack_int ldafb, lapack_int* ipiv, double* B, lapack_int ldb, double* X, lapack_int ldx, double* ferr, double* berr )
{
    return LAPACKE_dgbrfs( LAPACK_COL_MAJOR, trans, n, kl, ku, nrhs, AB, ldab, AFB, ldafb, ipiv, B, ldb, X, ldx, ferr, berr );
}

static lapack_int LAPACKE_gbrfs(
    char trans, lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs, std::complex<float>* AB, lapack_int ldab, std::complex<float>* AFB, lapack_int ldafb, lapack_int* ipiv, std::complex<float>* B, lapack_int ldb, std::complex<float>* X, lapack_int ldx, float* ferr, float* berr )
{
    return LAPACKE_cgbrfs( LAPACK_COL_MAJOR, trans, n, kl, ku, nrhs, AB, ldab, AFB, ldafb, ipiv, B, ldb, X, ldx, ferr, berr );
}

static lapack_int LAPACKE_gbrfs(
    char trans, lapack_int n, lapack_int kl, lapack_int ku, lapack_int nrhs, std::complex<double>* AB, lapack_int ldab, std::complex<double>* AFB, lapack_int ldafb, lapack_int* ipiv, std::complex<double>* B, lapack_int ldb, std::complex<double>* X, lapack_int ldx, double* ferr, double* berr )
{
    return LAPACKE_zgbrfs( LAPACK_COL_MAJOR, trans, n, kl, ku, nrhs, AB, ldab, AFB, ldafb, ipiv, B, ldb, X, ldx, ferr, berr );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gbrfs_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Op trans = params.trans.value();
    int64_t n = params.dim.n();
    int64_t kl = params.kl.value();
    int64_t ku = params.ku.value();
    int64_t nrhs = params.nrhs.value();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( kl+ku+1, align );
    int64_t ldafb = roundup( 2*kl*ku+1, align );
    int64_t ldb = roundup( max( 1, n ), align );
    int64_t ldx = roundup( max( 1, n ), align );
    size_t size_AB = (size_t) ldab * n;
    size_t size_AFB = (size_t) ldafb * n;
    size_t size_ipiv = (size_t) (n);
    size_t size_B = (size_t) ldb * nrhs;
    size_t size_X = (size_t) ldx * nrhs;
    size_t size_ferr = (size_t) (nrhs);
    size_t size_berr = (size_t) (nrhs);

    std::vector< scalar_t > AB( size_AB );
    std::vector< scalar_t > AFB( size_AFB );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );
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
    // todo: initialize ipiv_tst and ipiv_ref
    lapack::larnv( idist, iseed, B.size(), &B[0] );
    lapack::larnv( idist, iseed, X_tst.size(), &X_tst[0] );
    X_ref = X_tst;

    AFB = AB;
    int64_t info = lapack::gbtrf( n, n, kl, ku, &AFB[0], ldafb, &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::gbtrf returned error %lld\n", (lld) info );
    }

    info = lapack::gbtrs( trans, n, kl, ku, nrhs, &AFB[0], ldafb, &ipiv_tst[0], &B[0], ldb );
    if (info != 0) {
        fprintf( stderr, "lapack::gbtrs returned error %lld\n", (lld) info );
    }

    std::copy (ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::gbrfs( trans, n, kl, ku, nrhs, &AB[0], ldab, &AFB[0], ldafb, &ipiv_tst[0], &B[0], ldb, &X_tst[0], ldx, &ferr_tst[0], &berr_tst[0] );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gbrfs returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::gbrfs( trans, n, kl, ku, nrhs );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_gbrfs( op2char(trans), n, kl, ku, nrhs, &AB[0], ldab, &AFB[0], ldafb, &ipiv_ref[0], &B[0], ldb, &X_ref[0], ldx, &ferr_ref[0], &berr_ref[0] );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gbrfs returned error %lld\n", (lld) info_ref );
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
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gbrfs( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gbrfs_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gbrfs_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gbrfs_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gbrfs_work< std::complex<double> >( params, run );
            break;
    }
}
