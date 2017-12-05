#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_geqrf(
    lapack_int m, lapack_int n, float* A, lapack_int lda, float* tau )
{
    return LAPACKE_sgeqrf( LAPACK_COL_MAJOR, m, n, A, lda, tau );
}

static lapack_int LAPACKE_geqrf(
    lapack_int m, lapack_int n, double* A, lapack_int lda, double* tau )
{
    return LAPACKE_dgeqrf( LAPACK_COL_MAJOR, m, n, A, lda, tau );
}

static lapack_int LAPACKE_geqrf(
    lapack_int m, lapack_int n, std::complex<float>* A, lapack_int lda, std::complex<float>* tau )
{
    return LAPACKE_cgeqrf( LAPACK_COL_MAJOR, m, n, A, lda, tau );
}

static lapack_int LAPACKE_geqrf(
    lapack_int m, lapack_int n, std::complex<double>* A, lapack_int lda, std::complex<double>* tau )
{
    return LAPACKE_zgeqrf( LAPACK_COL_MAJOR, m, n, A, lda, tau );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_geqrf_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();

    // mark non-standard output values
    //params.ref_time.value();
    //params.ref_gflops.value();
    params.gflops.value();
    params.ortho.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, m ), align );
    size_t size_A = (size_t)( lda * n );
    size_t size_tau = (size_t)( min( m, n ) );
    int64_t minmn = min( m, n );

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > tau_tst( size_tau );
    std::vector< scalar_t > tau_ref( size_tau );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    A_ref = A_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::geqrf( m, n, &A_tst[0], lda, &tau_tst[0] );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::geqrf returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    double gflop = lapack::Gflop< scalar_t >::geqrf( m, n );
    params.gflops.value() = gflop / time;

    if (params.check.value() == 'y') {

        // ---------- check error
        // comparing to ref. solution doesn't work 
        // Following lapack/TESTING/LIN/zqrt01.f but using smaller Q and R
        real_t eps = std::numeric_limits< real_t >::epsilon();
        real_t tol = params.tol.value();

        int64_t ldq = m;
        std::vector< scalar_t > Q( m * minmn ); // m by k
        int64_t ldr = minmn;
        std::vector< scalar_t > R( minmn * n ); // k by n

        // Copy details of Q
        real_t rogue = -10000000000; // -1D+10
        lapack::laset( lapack::MatrixType::General, m, minmn, rogue, rogue, &Q[0], ldq );
        lapack::lacpy( lapack::MatrixType::Lower, m, minmn, &A_tst[0], lda, &Q[0], ldq );

        // Generate the m-by-m matrix Q
        int64_t info_ungqr = lapack::ungqr( m, minmn, minmn, &Q[0], ldq, &tau_tst[0] );
        if (info_ungqr != 0) {
            fprintf( stderr, "lapack::ungqr returned error %lld\n", (lld) info_ungqr );
        }

        // Copy R
        lapack::laset( lapack::MatrixType::Lower, minmn, n, 0.0, 0.0, &R[0], ldr );
        lapack::lacpy( lapack::MatrixType::Upper, minmn, n, &A_tst[0], lda, &R[0], ldr );

        // Compute R - Q'*A
        blas::gemm( Layout::ColMajor, Op::ConjTrans, Op::NoTrans, minmn, n, m, 
                    -1.0, &Q[0], ldq, &A_ref[0], lda, 1.0, &R[0], ldr );

        // Compute norm( R - Q'*A ) / ( M * norm(A) * EPS )
        real_t Anorm = lapack::lange( lapack::Norm::One, m, n, &A_ref[0], lda );
        real_t resid1 = lapack::lange( lapack::Norm::One, minmn, n, &R[0], ldr );
        real_t error1 = 0;
        if (Anorm > 0)
            error1 = resid1 / ( n * Anorm );

        // Compute I - Q'*Q
        lapack::laset( lapack::MatrixType::Upper, minmn, minmn, 0.0, 1.0, &R[0], ldr );
        blas::herk( Layout::ColMajor, Uplo::Upper, Op::ConjTrans, minmn, m, -1.0, &Q[0], ldq, 1.0, &R[0], ldr );

        // Compute norm( I - Q'*Q ) / ( M * EPS ) .
        real_t resid2 = lapack::lanhe( lapack::Norm::One, lapack::Uplo::Upper, minmn, &R[0], ldr );
        real_t error2 = ( resid2 / n );

        params.error.value() = error1;
        params.ortho.value() = error2;
        params.okay.value() = (error1 < tol*eps) && (error2 < tol*eps);
    }

    if (params.ref.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_geqrf( m, n, &A_ref[0], lda, &tau_ref[0] );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_geqrf returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( tau_tst, tau_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_geqrf( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_geqrf_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_geqrf_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_geqrf_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_geqrf_work< std::complex<double> >( params, run );
            break;
    }
}
