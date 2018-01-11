#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_geqlf(
    lapack_int m, lapack_int n, float* A, lapack_int lda, float* tau )
{
    return LAPACKE_sgeqlf( LAPACK_COL_MAJOR, m, n, A, lda, tau );
}

static lapack_int LAPACKE_geqlf(
    lapack_int m, lapack_int n, double* A, lapack_int lda, double* tau )
{
    return LAPACKE_dgeqlf( LAPACK_COL_MAJOR, m, n, A, lda, tau );
}

static lapack_int LAPACKE_geqlf(
    lapack_int m, lapack_int n, std::complex<float>* A, lapack_int lda, std::complex<float>* tau )
{
    return LAPACKE_cgeqlf( LAPACK_COL_MAJOR, m, n, A, lda, tau );
}

static lapack_int LAPACKE_geqlf(
    lapack_int m, lapack_int n, std::complex<double>* A, lapack_int lda, std::complex<double>* tau )
{
    return LAPACKE_zgeqlf( LAPACK_COL_MAJOR, m, n, A, lda, tau );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_geqlf_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ortho.value();
    params.time.value();
    params.gflops.value();
    params.ref_time.value();
    params.ref_gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, m ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_tau = (size_t) (min(m,n));

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
    double time = get_wtime();
    int64_t info_tst = lapack::geqlf( m, n, &A_tst[0], lda, &tau_tst[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::geqlf returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    double gflop = lapack::Gflop< scalar_t >::geqlf( m, n );
    params.gflops.value() = gflop / time;

    if (params.check.value() == 'y') {
        // ---------- check error
        // comparing to ref. solution doesn't work
        // Following magma/testing/testing_zgeqlf.cpp
        real_t eps = std::numeric_limits< real_t >::epsilon();
        real_t tol = params.tol.value();
        int64_t minmn = min( m, n );

        int64_t ldq = m;
        int64_t ldl = minmn;
        std::vector< scalar_t > Q( ldq * minmn ); // m by k
        std::vector< scalar_t > L( ldl * n ); // k by n

        // copy M by K matrix V to Q (copying diagonal, which isn't needed)
        // copy K by N matrix L
        lapack::laset( lapack::MatrixType::General, minmn, n, 0, 0, &L[0], ldl );
        if ( m >= n ) {
            int64_t m_n = m - n;
            lapack::lacpy( lapack::MatrixType::General, m_n, minmn, &A_tst[0], lda, &Q[0], ldq );
            lapack::lacpy( lapack::MatrixType::Upper, n, minmn, &A_tst[m_n], lda, &Q[m_n], ldq );
            lapack::lacpy( lapack::MatrixType::Lower, minmn, n, &A_tst[m_n], lda, &L[0], ldl );
        } else {
            int64_t n_m = n - m;
            lapack::lacpy( lapack::MatrixType::Upper, m, minmn, &A_tst[n_m*lda], lda, &Q[0], ldq );
            lapack::lacpy( lapack::MatrixType::General, minmn, n_m, &A_tst[0], lda, &L[0], ldl );
            lapack::lacpy( lapack::MatrixType::Lower, minmn, m, &A_tst[n_m*lda], lda, &L[n_m*ldl], ldl );
        }

        // generate M by K matrix Q, where K = min(M,N)
        int64_t info_ungql = lapack::ungql( m, minmn, minmn, &Q[0], ldq, &tau_tst[0] );
        if (info_ungql != 0) {
            fprintf( stderr, "lapack::ungqr returned error %lld\n", (lld) info_ungql );
        }

        // compute L - Q'*A
        blas::gemm( Layout::ColMajor, Op::ConjTrans, Op::NoTrans, minmn, n, m,
                    -1.0, &Q[0], ldq, &A_ref[0], lda, 1.0, &L[0], ldl );

        // error = || L - Q^H*A || / (N * ||A||)
        real_t Anorm = lapack::lange( lapack::Norm::One, m, n, &A_ref[0], lda );
        real_t resid1 = lapack::lange( lapack::Norm::One, minmn, n, &L[0], ldl );
        real_t error1 = 0;
        if (Anorm > 0)
            error1 = resid1 / ( n * Anorm );

        // set L = I (K by K identity), then L = I - Q^H*Q
        lapack::laset( lapack::MatrixType::Upper, minmn, minmn, 0.0, 1.0, &L[0], ldl );
        blas::herk( Layout::ColMajor, Uplo::Upper, Op::ConjTrans, minmn, m, -1.0, &Q[0], ldq, 1.0, &L[0], ldl );

        // error = || I - Q^H*Q || / N
        real_t resid2 = lapack::lanhe( lapack::Norm::One, lapack::Uplo::Upper, minmn, &L[0], ldl );
        real_t error2 = ( resid2 / n );

        params.error.value() = error1;
        params.ortho.value() = error2;
        params.okay.value() = (error1 < tol*eps) && (error2 < tol*eps);
    }

    if (params.ref.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_geqlf( m, n, &A_ref[0], lda, &tau_ref[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_geqlf returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        params.ref_gflops.value() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_geqlf( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_geqlf_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_geqlf_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_geqlf_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_geqlf_work< std::complex<double> >( params, run );
            break;
    }
}
