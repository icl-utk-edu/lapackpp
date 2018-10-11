#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_lansp_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Norm norm = params.norm();
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();

    if (! run)
        return;

    // ---------- setup
    size_t size_AP = (size_t) (n*(n+1)/2);

    std::vector< scalar_t > AP( size_AP );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AP.size(), &AP[0] );

    if (verbose >= 2) {
        printf( "AP = " ); print_vector( AP.size(), &AP[0], 1 );
    }

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = get_wtime();
    real_t norm_tst = lapack::lansp( norm, uplo, n, &AP[0] );
    time = get_wtime() - time;

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::lansp( norm, n );
    //params.gflops() = gflop / time;

    if (verbose >= 1) {
        printf( "norm_tst = %.8e\n", norm_tst );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = get_wtime();
        real_t norm_ref = LAPACKE_lansp( norm2char(norm), uplo2char(uplo), n, &AP[0] );
        time = get_wtime() - time;

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;

        if (verbose >= 1) {
            printf( "norm_ref = %.8e\n", norm_ref );
        }

        // ---------- check error compared to reference
        real_t error = 0;
        error += std::abs( norm_tst - norm_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_lansp( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_lansp_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_lansp_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_lansp_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_lansp_work< std::complex<double> >( params, run );
            break;
    }
}
