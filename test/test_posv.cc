#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_posv_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs();
    int64_t align = params.align();
    int64_t verbose = params.verbose();
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();
    params.gflops();

    if (! run) {
        params.matrix.kind.set_default( "rand_dominant" );
        return;
    }

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, n ), align );
    int64_t ldb = roundup( blas::max( 1, n ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_B = (size_t) ldb * nrhs;

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );

    lapack::generate_matrix( params.matrix, n, n, &A_tst[0], lda );
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );

    A_ref = A_tst;
    B_ref = B_tst;

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld\n"
                "B n=%5lld, nrhs=%5lld, ldb=%5lld\n",
                (lld) n, (lld) lda,
                (lld) n, (lld) nrhs, (lld) ldb );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( n, n, &A_tst[0], lda );
        printf( "B = " ); print_matrix( n, nrhs, &B_tst[0], lda );
    }

    // test error exits
    if (params.error_exit() == 'y') {
        using lapack::Uplo;
        assert_throw( lapack::posv( Uplo(0),  n, nrhs, &A_tst[0], lda, &B_tst[0], ldb ), lapack::Error );
        assert_throw( lapack::posv( uplo,    -1, nrhs, &A_tst[0], lda, &B_tst[0], ldb ), lapack::Error );
        assert_throw( lapack::posv( uplo,     n,   -1, &A_tst[0], lda, &B_tst[0], ldb ), lapack::Error );
        assert_throw( lapack::posv( uplo,     n, nrhs, &A_tst[0], n-1, &B_tst[0], ldb ), lapack::Error );
        assert_throw( lapack::posv( uplo,     n, nrhs, &A_tst[0], lda, &B_tst[0], n-1 ), lapack::Error );
    }

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = libtest::get_wtime();
    int64_t info_tst = lapack::posv( uplo, n, nrhs, &A_tst[0], lda, &B_tst[0], ldb );
    time = libtest::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::posv returned error %lld\n", (lld) info_tst );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::posv( n, nrhs );
    params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "A2 = " ); print_matrix( n, n, &A_tst[0], lda );
        printf( "B2 = " ); print_matrix( n, nrhs, &B_tst[0], ldb );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = libtest::get_wtime();
        int64_t info_ref = LAPACKE_posv( uplo2char(uplo), n, nrhs, &A_ref[0], lda, &B_ref[0], ldb );
        time = libtest::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_posv returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "A2ref = " ); print_matrix( n, n, &A_ref[0], lda );
            printf( "B2ref = " ); print_matrix( n, nrhs, &B_ref[0], ldb );
        }

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( B_tst, B_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_posv( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_posv_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_posv_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_posv_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_posv_work< std::complex<double> >( params, run );
            break;
    }
}
