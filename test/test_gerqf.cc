#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gerqf_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();
    params.matrix.mark();

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
    size_t size_tau = (size_t) ( min( m, n ) );

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > tau_tst( size_tau );
    std::vector< scalar_t > tau_ref( size_tau );

    lapack::generate_matrix( params.matrix, m, n, nullptr, &A_tst[0], lda );
    A_ref = A_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::gerqf( m, n, &A_tst[0], lda, &tau_tst[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gerqf returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    double gflop = lapack::Gflop< scalar_t >::gerqf( m, n );
    params.gflops.value() = gflop / time;

    if (params.check.value() == 'y') {
        // ---------- check error
        // comparing to ref. solution doesn't work
        // Following lapack/TESTING/LIN/crqt01.f
        real_t eps = std::numeric_limits< real_t >::epsilon();
        real_t tol = params.tol.value();
        int64_t minmn = min( m, n );
        int64_t maxmn = max( m, n );
        int64_t m_n = m - n;
        int64_t n_m = n - m;

        int64_t ldq = maxmn;
        std::vector< scalar_t > Q( ldq * n ); // n-by-n orthogonal matrix Q.
        int64_t ldr = maxmn;
        std::vector< scalar_t > R( ldr * maxmn );

        // Copy details of Q
        lapack::laset( lapack::MatrixType::General, n, n, -10000000000, -1000000000, &Q[0], ldq );
        if (m <= n) {
            if (m < n)
                lapack::lacpy( lapack::MatrixType::General, m, n_m, &A_tst[0], lda, &Q[n_m], ldq );
            lapack::lacpy( lapack::MatrixType::Lower, m-1, m-1, &A_tst[1+(n_m*lda)], lda, &Q[n_m+1+(n_m*ldq)], ldq );
        } else {
            lapack::lacpy( lapack::MatrixType::Lower, n-1, n-1, &A_tst[m_n+1], lda, &Q[1], ldq );
        }

        // Generate the n-by-n matrix Q
        int64_t info_ungrq = lapack::ungrq( n, n, minmn, &Q[0], ldq, &tau_tst[0] );
        if (info_ungrq != 0) {
            fprintf( stderr, "lapack::ungqr returned error %lld\n", (lld) info_ungrq );
        }

        // Copy R
        lapack::laset( lapack::MatrixType::General, m, n, 0, 0, &R[0], ldr );
        if ( m <= n ) {
            lapack::lacpy( lapack::MatrixType::Upper, m, m, &A_tst[n_m*lda], lda, &R[n_m*ldr], ldr );
        } else {
            lapack::lacpy( lapack::MatrixType::General, m_n, n, &A_tst[0], lda, &R[0], ldr );
            lapack::lacpy( lapack::MatrixType::Upper, n, n, &A_tst[m_n], lda, &R[m_n], ldr );
        }

        // Compute R - A*Q'
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::ConjTrans, m, n, n,
                    -1.0, &A_ref[0], lda, &Q[0], ldq, 1.0, &R[0], ldr );

        // error = || L - Q^H*A || / (N * ||A||)
        real_t Anorm = lapack::lange( lapack::Norm::One, m, n, &A_ref[0], lda );
        real_t resid1 = lapack::lange( lapack::Norm::One, m, n, &R[0], ldr );
        real_t error1 = 0;
        if (Anorm > 0)
            error1 = resid1 / ( n * Anorm );

        // Compute I - Q*Q'
        lapack::laset( lapack::MatrixType::General, n, n, 0.0, 1.0, &R[0], ldr );
        blas::herk( Layout::ColMajor, Uplo::Upper, Op::NoTrans, n, n, -1.0, &Q[0], ldq, 1.0, &R[0], ldr );

        // error = || I - Q^H*Q || / N
        real_t resid2 = lapack::lanhe( lapack::Norm::One, lapack::Uplo::Upper, n, &R[0], ldr );
        real_t error2 = ( resid2 / n );

        params.error.value() = error1;
        params.ortho.value() = error2;
        params.okay.value() = (error1 < tol*eps) && (error2 < tol*eps);
    }

    if (params.ref.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_gerqf( m, n, &A_ref[0], lda, &tau_ref[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gerqf returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        params.ref_gflops.value() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_gerqf( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gerqf_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gerqf_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gerqf_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gerqf_work< std::complex<double> >( params, run );
            break;
    }
}
