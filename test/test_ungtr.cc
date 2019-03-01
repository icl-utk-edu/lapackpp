#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_ungtr_work( Params& params, bool run )
{
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
    int64_t lda = roundup( n, align );
    size_t size_A = (size_t) lda * n;
    size_t size_tau = (size_t) (n-1);
    size_t size_D = (size_t) (n);
    size_t size_E = (size_t) (n-1);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > tau( size_tau );
    std::vector< real_t > D_tst( size_D );
    std::vector< real_t > E_tst( size_E );

    lapack::generate_matrix( params.matrix, n, n, &A_tst[0], lda );

    // reduce to tridiagonal form to use the tau later
    int64_t info = lapack::hetrd( uplo, n, &A_tst[0], lda, &D_tst[0], &E_tst[0], &tau[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::gttrf returned error %lld\n", (lld) info );
    }

    // save A_tst for reference run
    A_ref = A_tst;

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = libtest::get_wtime();
    int64_t info_tst = lapack::ungtr( uplo, n, &A_tst[0], lda, &tau[0] );
    time = libtest::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::ungtr returned error %lld\n", (lld) info_tst );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::ungtr( n );
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = libtest::get_wtime();
        int64_t info_ref = LAPACKE_ungtr( uplo2char(uplo), n, &A_ref[0], lda, &tau[0] );
        time = libtest::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_ungtr returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

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
void test_ungtr( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_ungtr_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_ungtr_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_ungtr_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_ungtr_work< std::complex<double> >( params, run );
            break;
    }
}
