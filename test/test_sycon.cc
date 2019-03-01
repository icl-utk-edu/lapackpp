#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_sycon_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t align = params.align();
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( blas::max( 1, n ), align );
    real_t anorm = 0;  // todo value
    real_t rcond_tst;
    real_t rcond_ref;
    size_t size_A = (size_t) lda * n;
    size_t size_ipiv = (size_t) (n);

    std::vector< scalar_t > A( size_A );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );

    lapack::generate_matrix( params.matrix, n, n, &A[0], lda );

    // ---------- factor before test
    int64_t info = lapack::sytrf( uplo, n, &A[0], lda, &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::sytrf returned error %lld\n", (lld) info );
    }
    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = libtest::get_wtime();
    int64_t info_tst = lapack::sycon( uplo, n, &A[0], lda, &ipiv_tst[0], anorm, &rcond_tst );
    time = libtest::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::sycon returned error %lld\n", (lld) info_tst );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::sycon( n );
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- reuse factorization and initialize ipiv_ref
        std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = libtest::get_wtime();
        int64_t info_ref = LAPACKE_sycon( uplo2char(uplo), n, &A[0], lda, &ipiv_ref[0], anorm, &rcond_ref );
        time = libtest::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_sycon returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += std::abs( rcond_tst - rcond_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_sycon( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_sycon_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_sycon_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_sycon_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_sycon_work< std::complex<double> >( params, run );
            break;
    }
}
