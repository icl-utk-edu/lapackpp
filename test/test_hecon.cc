#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_hecon_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
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
    int64_t lda = roundup( max( 1, n ), align );
    real_t anorm;
    real_t rcond_tst;
    real_t rcond_ref;
    size_t size_A = (size_t) lda * n;
    size_t size_ipiv = (size_t) (n);

    std::vector< scalar_t > A( size_A );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );

    lapack::generate_matrix( params.matrix, n, n, &A[0], lda );

    // initialize anorm as the norm of the original matrix
    anorm = lapack::lange( lapack::Norm::One, n, n, &A[0], lda );

    // factor A to initialize ipiv_tst and ipiv_ref
    int64_t info_trf = lapack::hetrf( uplo, n, &A[0], lda, &ipiv_tst[0] );
    if (info_trf != 0) {
        fprintf( stderr, "lapack::hetrf returned error %lld\n", (lld) info_trf );
    }
    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = get_wtime();
    int64_t info_tst = lapack::hecon( uplo, n, &A[0], lda, &ipiv_tst[0], anorm, &rcond_tst );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::hecon returned error %lld\n", (lld) info_tst );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::hecon( n );
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_hecon( uplo2char(uplo), n, &A[0], lda, &ipiv_ref[0], anorm, &rcond_ref );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_hecon returned error %lld\n", (lld) info_ref );
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
void test_hecon( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_hecon_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_hecon_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_hecon_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_hecon_work< std::complex<double> >( params, run );
            break;
    }
}
