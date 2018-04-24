#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_getri_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t align = params.align.value();
    int64_t verbose = params.verbose.value();
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();
    params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_ipiv = (size_t) (n);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );

    lapack::generate_matrix( params.matrix, n, n, nullptr, &A_tst[0], lda );
    A_ref = A_tst;

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld\n",
                (lld) n, (lld) lda );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( n, n, &A_tst[0], lda );
    }

    // factor A into LU
    int64_t info = lapack::getrf( n, n, &A_tst[0], lda, &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::getrf returned error %lld\n", (lld) info );
    }

    // test error exits
    if (params.error_exit.value() == 'y') {
        assert_throw( lapack::getri( -1, &A_tst[0], lda, &ipiv_tst[0] ), lapack::Error );
        assert_throw( lapack::getri(  n, &A_tst[0], n-1, &ipiv_tst[0] ), lapack::Error );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::getri( n, &A_tst[0], lda, &ipiv_tst[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::getri returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    double gflop = lapack::Gflop< scalar_t >::getri( n );
    params.gflops.value() = gflop / time;

    if (verbose >= 2) {
        printf( "A2 = " ); print_matrix( n, n, &A_tst[0], lda );
    }

    if (params.check.value() == 'y') {
        // ---------- check error
        // comparing to ref. solution doesn't work due to roundoff errors
        real_t eps = std::numeric_limits< real_t >::epsilon();
        real_t tol = params.tol.value();

        // R = I
        std::vector< scalar_t > R( size_A );
        // todo: laset; needs uplo=general
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < n; ++i) {
                R[ i + j*lda ] = 0;
            }
            R[ j + j*lda ] = 1;
        }

        // R = I - A A^{-1}
        blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, n, n,
                    -1.0, &A_ref[0], lda,
                          &A_tst[0], lda,
                     1.0, &R[0], lda );
        if (verbose >= 2) {
            printf( "R = " ); print_matrix( n, n, &R[0], lda );
        }

        // error = ||I - A A^{-1}|| / (n ||A|| ||A^{-1}||)
        real_t Rnorm     = lapack::lange( lapack::Norm::Fro, n, n, &R[0],     lda );
        real_t Anorm     = lapack::lange( lapack::Norm::Fro, n, n, &A_ref[0], lda );
        real_t Ainv_norm = lapack::lange( lapack::Norm::Fro, n, n, &A_tst[0], lda );
        real_t error = Rnorm / (n * Anorm * Ainv_norm);
        params.error.value() = error;
        params.okay.value() = (error < tol*eps);
    }

    if (params.ref.value() == 'y') {
        // factor A into LU
        info = LAPACKE_getrf( n, n, &A_ref[0], lda, &ipiv_ref[0] );
        if (info != 0) {
            fprintf( stderr, "LAPACKE_getrf returned error %lld\n", (lld) info );
        }

        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_getri( n, &A_ref[0], lda, &ipiv_ref[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_getri returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        params.ref_gflops.value() = gflop / time;

        if (verbose >= 2) {
            printf( "A2ref = " ); print_matrix( n, n, &A_ref[0], lda );
        }
    }
}

// -----------------------------------------------------------------------------
void test_getri( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_getri_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_getri_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_getri_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_getri_work< std::complex<double> >( params, run );
            break;
    }
}
