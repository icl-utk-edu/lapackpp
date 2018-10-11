#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gelqf_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align();
    params.matrix.mark();

    // mark non-standard output values
    params.ortho();
    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, m ), align );
    int64_t minmn = min( m, n );
    size_t size_A = (size_t)(lda * n);
    size_t size_tau = (size_t)(min(m,n));

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > tau_tst( size_tau );
    std::vector< scalar_t > tau_ref( size_tau );

    lapack::generate_matrix( params.matrix, m, n, &A_tst[0], lda );
    A_ref = A_tst;

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = get_wtime();
    int64_t info_tst = lapack::gelqf( m, n, &A_tst[0], lda, &tau_tst[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gelqf returned error %lld\n", (lld) info_tst );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::gelqf( m, n );
    params.gflops() = gflop / time;

    if (params.check() == 'y') {
        // ---------- check error
        // comparing to ref. solution doesn't work
        // Taken from lapack/TESTING/LIN/zlqt01.f but using smaller Q and L
        real_t eps = std::numeric_limits< real_t >::epsilon();
        real_t tol = params.tol();

        int64_t ldq = minmn;
        std::vector< scalar_t > Q( minmn * n );
        int64_t ldl = m;
        std::vector< scalar_t > L( m * minmn );

        // Copy details of Q
        real_t rogue = -10000000000; // -1D+10
        lapack::laset( lapack::MatrixType::General, minmn, n, rogue, rogue, &Q[0], ldq );
        if (n > 1)
            lapack::lacpy( lapack::MatrixType::Upper, minmn, n, &A_tst[0], lda, &Q[0], ldq );

        // Generate the m-by-m matrix Q
        int64_t info_unglq = lapack::unglq( minmn, n, minmn, &Q[0], ldq, &tau_tst[0] );
        if (info_unglq != 0) {
            fprintf( stderr, "lapack::unglq returned error %lld\n", (lld) info_unglq );
        }

        // Copy L
        lapack::laset( lapack::MatrixType::Upper, m, minmn, 0.0, 0.0, &L[0], ldl );
        lapack::lacpy( lapack::MatrixType::Lower, m, minmn, &A_tst[0], lda, &L[0], ldl );

        // Compute L - A*Q'
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::ConjTrans, m, minmn, n,
                    -1.0, &A_ref[0], lda, &Q[0], ldq, 1.0, &L[0], ldl );

        // Compute norm( L - Q'*A ) / ( N * norm(A) * EPS ) .
        real_t Anorm = lapack::lange( lapack::Norm::One, m, n, &A_ref[0], lda );
        real_t resid1 = lapack::lange( lapack::Norm::One, m, minmn, &L[0], ldl );
        real_t error1 = 0;
        if (Anorm > 0)
            error1 = ( resid1 / ( n * Anorm ) );

        // Compute I - Q*Q'
        lapack::laset( lapack::MatrixType::Upper, minmn, minmn, 0.0, 1.0, &L[0], ldl );
        blas::herk( Layout::ColMajor, Uplo::Upper, Op::NoTrans, minmn, n, -1.0, &Q[0], ldq, 1.0, &L[0], ldl );

        // Compute norm( I - Q*Q' ) / ( N * EPS ) .
        real_t resid2 = lapack::lanhe( lapack::Norm::One, lapack::Uplo::Upper, minmn, &L[0], ldl );
        real_t error2 = ( resid2 / n );

        params.error() = error1;
        params.ortho() = error2;
        params.okay() = ( error1 < tol*eps ) && ( error2 < tol*eps );
    }

    if (params.ref() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_gelqf( m, n, &A_ref[0], lda, &tau_ref[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gelqf returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_gelqf( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gelqf_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gelqf_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gelqf_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gelqf_work< std::complex<double> >( params, run );
            break;
    }
}
