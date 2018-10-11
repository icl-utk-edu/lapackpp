#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_langt_work( Params& params, bool run )
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
    size_t size_DL = (size_t) (n-1);
    size_t size_D = (size_t) (n);
    size_t size_DU = (size_t) (n-1);

    std::vector< scalar_t > DL( size_DL );
    std::vector< scalar_t > D( size_D );
    std::vector< scalar_t > DU( size_DU );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, DL.size(), &DL[0] );
    lapack::larnv( idist, iseed, D.size(), &D[0] );
    lapack::larnv( idist, iseed, DU.size(), &DU[0] );

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = get_wtime();
    int64_t norm_tst = lapack::langt( norm, n, &DL[0], &D[0], &DU[0] );
    time = get_wtime() - time;

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::langt( norm, n );
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = get_wtime();
        int64_t norm_ref = LAPACKE_langt( norm2char(norm), n, &DL[0], &D[0], &DU[0] );
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
void test_langt( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_langt_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_langt_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_langt_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_langt_work< std::complex<double> >( params, run );
            break;
    }
}
