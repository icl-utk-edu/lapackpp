#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_potrf_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
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
    size_t size_A = (size_t) lda * n;

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );

    lapack::generate_matrix( params.matrix, n, n, &A_tst[0], lda );
    A_ref = A_tst;

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld\n",
                (lld) n, (lld) lda );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( n, n, &A_tst[0], lda );
    }

    // test error exits
    if (params.error_exit() == 'y') {
        using lapack::Uplo;
        assert_throw( lapack::potrf( Uplo(0),  n, &A_tst[0], lda ), lapack::Error );
        assert_throw( lapack::potrf( uplo,    -1, &A_tst[0], lda ), lapack::Error );
        assert_throw( lapack::potrf( uplo,     n, &A_tst[0], n-1 ), lapack::Error );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::potrf( uplo, n, &A_tst[0], lda );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::potrf returned error %lld\n", (lld) info_tst );
    }

    params.time() = time;
    double gflop = lapack::Gflop< scalar_t >::potrf( n );
    params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "A2 = " ); print_matrix( n, n, &A_tst[0], lda );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_potrf( uplo2char(uplo), n, &A_ref[0], lda );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_potrf returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "A2ref = " ); print_matrix( n, n, &A_ref[0], lda );
        }

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_potrf( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_potrf_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_potrf_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_potrf_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_potrf_work< std::complex<double> >( params, run );
            break;
    }
}
