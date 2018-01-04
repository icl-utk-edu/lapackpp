#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gbcon(
    char norm, lapack_int n, lapack_int kl, lapack_int ku, float* AB, lapack_int ldab, lapack_int* ipiv, float anorm, float* rcond )
{
    return LAPACKE_sgbcon( LAPACK_COL_MAJOR, norm, n, kl, ku, AB, ldab, ipiv, anorm, rcond );
}

static lapack_int LAPACKE_gbcon(
    char norm, lapack_int n, lapack_int kl, lapack_int ku, double* AB, lapack_int ldab, lapack_int* ipiv, double anorm, double* rcond )
{
    return LAPACKE_dgbcon( LAPACK_COL_MAJOR, norm, n, kl, ku, AB, ldab, ipiv, anorm, rcond );
}

static lapack_int LAPACKE_gbcon(
    char norm, lapack_int n, lapack_int kl, lapack_int ku, std::complex<float>* AB, lapack_int ldab, lapack_int* ipiv, float anorm, float* rcond )
{
    return LAPACKE_cgbcon( LAPACK_COL_MAJOR, norm, n, kl, ku, AB, ldab, ipiv, anorm, rcond );
}

static lapack_int LAPACKE_gbcon(
    char norm, lapack_int n, lapack_int kl, lapack_int ku, std::complex<double>* AB, lapack_int ldab, lapack_int* ipiv, double anorm, double* rcond )
{
    return LAPACKE_zgbcon( LAPACK_COL_MAJOR, norm, n, kl, ku, AB, ldab, ipiv, anorm, rcond );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gbcon_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Norm norm = params.norm.value();
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
    real_t anorm;
    real_t rcond_tst;
    real_t rcond_ref;
    size_t size_AB = (size_t) ldab * n;
    size_t size_ipiv = (size_t) (n);

    std::vector< scalar_t > AB( size_AB );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB.size(), &AB[0] );

    anorm = lapack::langb( norm, n, kl, ku, &AB[kl], ldab );

    int64_t info = lapack::gbtrf( n, n, kl, ku, &AB[0], ldab, &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::gbtrf returned error &lld\n", (lld) info );
    }
    
    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::gbcon( norm, n, kl, ku, &AB[0], ldab, &ipiv_tst[0], anorm, &rcond_tst );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gbcon returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::gbcon( norm, n, kl, ku );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_gbcon( norm2char(norm), n, kl, ku, &AB[0], ldab, &ipiv_ref[0], anorm, &rcond_ref );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gbcon returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += std::abs( rcond_tst - rcond_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gbcon( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gbcon_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gbcon_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gbcon_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gbcon_work< std::complex<double> >( params, run );
            break;
    }
}
