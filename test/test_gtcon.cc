#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gtcon(
    char norm, lapack_int n, float* DL, float* D, float* DU, float* DU2, lapack_int* ipiv, float anorm, float* rcond )
{
    //return LAPACKE_sgtcon( LAPACK_COL_MAJOR, norm, n, DL, D, DU, DU2, ipiv, anorm, rcond );
    return LAPACKE_sgtcon( norm, n, DL, D, DU, DU2, ipiv, anorm, rcond );
}

static lapack_int LAPACKE_gtcon(
    char norm, lapack_int n, double* DL, double* D, double* DU, double* DU2, lapack_int* ipiv, double anorm, double* rcond )
{
    //return LAPACKE_dgtcon( LAPACK_COL_MAJOR, norm, n, DL, D, DU, DU2, ipiv, anorm, rcond );
    return LAPACKE_dgtcon( norm, n, DL, D, DU, DU2, ipiv, anorm, rcond );
}

static lapack_int LAPACKE_gtcon(
    char norm, lapack_int n, std::complex<float>* DL, std::complex<float>* D, std::complex<float>* DU, std::complex<float>* DU2, lapack_int* ipiv, float anorm, float* rcond )
{
    //return LAPACKE_cgtcon( LAPACK_COL_MAJOR, norm, n, DL, D, DU, DU2, ipiv, anorm, rcond );
    return LAPACKE_cgtcon( norm, n, DL, D, DU, DU2, ipiv, anorm, rcond );
}

static lapack_int LAPACKE_gtcon(
    char norm, lapack_int n, std::complex<double>* DL, std::complex<double>* D, std::complex<double>* DU, std::complex<double>* DU2, lapack_int* ipiv, double anorm, double* rcond )
{
    //return LAPACKE_zgtcon( LAPACK_COL_MAJOR, norm, n, DL, D, DU, DU2, ipiv, anorm, rcond );
    return LAPACKE_zgtcon( norm, n, DL, D, DU, DU2, ipiv, anorm, rcond );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gtcon_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Norm norm = params.norm.value();
    int64_t n = params.dim.n();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    real_t anorm;  // todo value
    real_t rcond_tst;
    real_t rcond_ref;
    size_t size_DL = (size_t) (n-1);
    size_t size_D = (size_t) (n);
    size_t size_DU = (size_t) (n-1);
    size_t size_DU2 = (size_t) (n-2);
    size_t size_ipiv = (size_t) (n);

    std::vector< scalar_t > DL( size_DL );
    std::vector< scalar_t > D( size_D );
    std::vector< scalar_t > DU( size_DU );
    std::vector< scalar_t > DU2( size_DU2 );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, DL.size(), &DL[0] );
    lapack::larnv( idist, iseed, D.size(), &D[0] );
    lapack::larnv( idist, iseed, DU.size(), &DU[0] );
    lapack::larnv( idist, iseed, DU2.size(), &DU2[0] );
    // todo: initialize ipiv_tst and ipiv_ref

    // compute norm
    anorm = lapack::langt( norm, n, &DL[0], &D[0], &DU[0] );

    // factor
    int64_t info = lapack::gttrf( n, &DL[0], &D[0], &DU[0], &DU2[0], &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::gttrf returned error %lld\n", (lld) info );
    }

    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::gtcon( norm, n, &DL[0], &D[0], &DU[0], &DU2[0], &ipiv_tst[0], anorm, &rcond_tst );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gtcon returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::gtcon( norm, n );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_gtcon( norm2char(norm), n, &DL[0], &D[0], &DU[0], &DU2[0], &ipiv_ref[0], anorm, &rcond_ref );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gtcon returned error %lld\n", (lld) info_ref );
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
void test_gtcon( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gtcon_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gtcon_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gtcon_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gtcon_work< std::complex<double> >( params, run );
            break;
    }
}
