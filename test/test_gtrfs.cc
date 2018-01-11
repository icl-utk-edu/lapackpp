#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gtrfs(
    char trans, lapack_int n, lapack_int nrhs, float* DL, float* D, float* DU, float* DLF, float* DF, float* DUF, float* DU2, lapack_int* ipiv, float* B, lapack_int ldb, float* X, lapack_int ldx, float* ferr, float* berr )
{
    return LAPACKE_sgtrfs( LAPACK_COL_MAJOR, trans, n, nrhs, DL, D, DU, DLF, DF, DUF, DU2, ipiv, B, ldb, X, ldx, ferr, berr );
}

static lapack_int LAPACKE_gtrfs(
    char trans, lapack_int n, lapack_int nrhs, double* DL, double* D, double* DU, double* DLF, double* DF, double* DUF, double* DU2, lapack_int* ipiv, double* B, lapack_int ldb, double* X, lapack_int ldx, double* ferr, double* berr )
{
    return LAPACKE_dgtrfs( LAPACK_COL_MAJOR, trans, n, nrhs, DL, D, DU, DLF, DF, DUF, DU2, ipiv, B, ldb, X, ldx, ferr, berr );
}

static lapack_int LAPACKE_gtrfs(
    char trans, lapack_int n, lapack_int nrhs, std::complex<float>* DL, std::complex<float>* D, std::complex<float>* DU, std::complex<float>* DLF, std::complex<float>* DF, std::complex<float>* DUF, std::complex<float>* DU2, lapack_int* ipiv, std::complex<float>* B, lapack_int ldb, std::complex<float>* X, lapack_int ldx, float* ferr, float* berr )
{
    return LAPACKE_cgtrfs( LAPACK_COL_MAJOR, trans, n, nrhs, DL, D, DU, DLF, DF, DUF, DU2, ipiv, B, ldb, X, ldx, ferr, berr );
}

static lapack_int LAPACKE_gtrfs(
    char trans, lapack_int n, lapack_int nrhs, std::complex<double>* DL, std::complex<double>* D, std::complex<double>* DU, std::complex<double>* DLF, std::complex<double>* DF, std::complex<double>* DUF, std::complex<double>* DU2, lapack_int* ipiv, std::complex<double>* B, lapack_int ldb, std::complex<double>* X, lapack_int ldx, double* ferr, double* berr )
{
    return LAPACKE_zgtrfs( LAPACK_COL_MAJOR, trans, n, nrhs, DL, D, DU, DLF, DF, DUF, DU2, ipiv, B, ldb, X, ldx, ferr, berr );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gtrfs_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Op trans = params.trans.value();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs.value();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t ldb = roundup( max( 1, n ), align );
    int64_t ldx = roundup( max( 1, n ), align );
    size_t size_DL = (size_t) (n-1);
    size_t size_D = (size_t) (n);
    size_t size_DU = (size_t) (n-1);
    size_t size_DLF = (size_t) (n-1);
    size_t size_DF = (size_t) (n);
    size_t size_DUF = (size_t) (n-1);
    size_t size_DU2 = (size_t) (n-2);
    size_t size_ipiv = (size_t) (n);
    size_t size_B = (size_t) ldb * nrhs;
    size_t size_X = (size_t) ldx * nrhs;
    size_t size_ferr = (size_t) (nrhs);
    size_t size_berr = (size_t) (nrhs);

    std::vector< scalar_t > DL( size_DL );
    std::vector< scalar_t > D( size_D );
    std::vector< scalar_t > DU( size_DU );
    std::vector< scalar_t > DLF( size_DLF );
    std::vector< scalar_t > DF( size_DF );
    std::vector< scalar_t > DUF( size_DUF );
    std::vector< scalar_t > DU2( size_DU2 );
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
    lapack::larnv( idist, iseed, DL.size(), &DL[0] );
    lapack::larnv( idist, iseed, D.size(), &D[0] );
    lapack::larnv( idist, iseed, DU.size(), &DU[0] );
    lapack::larnv( idist, iseed, DLF.size(), &DLF[0] );
    lapack::larnv( idist, iseed, DF.size(), &DF[0] );
    lapack::larnv( idist, iseed, DUF.size(), &DUF[0] );
    lapack::larnv( idist, iseed, DU2.size(), &DU2[0] );
    lapack::larnv( idist, iseed, B.size(), &B[0] );
    lapack::larnv( idist, iseed, X_tst.size(), &X_tst[0] );
    X_ref = X_tst;

    // condition to be diagonally dominate
    for (int64_t i = 0; i < n; ++i) {
        D[i] += 4;
    }

    // factor using gttrf
    int64_t info = lapack::gttrf( n, &DL[0], &D[0], &DU[0], &DU2[0], &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::gttrf returned error %lld\n", (lld) info );
    }
    
    // solve using gttrf
    info = lapack::gttrf( n, &DL[0], &D[0], &DU[0], &DU2[0], &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::gttrf returned error %lld\n", (lld) info );
    }

    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::gtrfs( trans, n, nrhs, &DL[0], &D[0], &DU[0], &DLF[0], &DF[0], &DUF[0], &DU2[0], &ipiv_tst[0], &B[0], ldb, &X_tst[0], ldx, &ferr_tst[0], &berr_tst[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gtrfs returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::gtrfs( trans, n, nrhs );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_gtrfs( op2char(trans), n, nrhs, &DL[0], &D[0], &DU[0], &DLF[0], &DF[0], &DUF[0], &DU2[0], &ipiv_ref[0], &B[0], ldb, &X_ref[0], ldx, &ferr_ref[0], &berr_ref[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gtrfs returned error %lld\n", (lld) info_ref );
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
void test_gtrfs( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gtrfs_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gtrfs_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gtrfs_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gtrfs_work< std::complex<double> >( params, run );
            break;
    }
}
