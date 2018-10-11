#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_lantp_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Norm norm = params.norm();
    lapack::Uplo uplo = params.uplo();
    lapack::Diag diag = params.diag();
    int64_t n = params.dim.n();

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();

    if (! run)
        return;

    // ---------- setup
    size_t size_AP = (size_t) (n*(n+1)/2);

    std::vector< scalar_t > AP( size_AP );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AP.size(), &AP[0] );

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = get_wtime();
    int64_t norm_tst = lapack::lantp( norm, uplo, diag, n, &AP[0] );
    time = get_wtime() - time;

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::lantp( norm, diag, n );
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = get_wtime();
        int64_t norm_ref = LAPACKE_lantp( norm2char(norm), uplo2char(uplo), diag2char(diag), n, &AP[0] );
        time = get_wtime() - time;

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (norm_tst != norm_ref) {
            error = 1;
        }
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_lantp( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_lantp_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_lantp_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_lantp_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_lantp_work< std::complex<double> >( params, run );
            break;
    }
}
