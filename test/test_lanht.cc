#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_lanht_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Norm norm = params.norm();
    int64_t n = params.dim.n();

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();

    if (! run)
        return;

    // ---------- setup
    size_t size_D = (size_t) (n);
    size_t size_E = (size_t) (n-1);

    std::vector< real_t > D( size_D );
    std::vector< scalar_t > E( size_E );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, D.size(), &D[0] );
    lapack::larnv( idist, iseed, E.size(), &E[0] );

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = get_wtime();
    int64_t norm_tst = lapack::lanht( norm, n, &D[0], &E[0] );
    time = get_wtime() - time;

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::lanht( norm, n );
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = get_wtime();
        int64_t norm_ref = LAPACKE_lanht( norm2char(norm), n, &D[0], &E[0] );
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
void test_lanht( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_lanht_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_lanht_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_lanht_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_lanht_work< std::complex<double> >( params, run );
            break;
    }
}
