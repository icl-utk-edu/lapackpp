#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_lanhb_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Norm norm = params.norm();
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t kd = min( params.kd(), n-1 );
    int64_t align = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( kd+1, align );
    size_t size_AB = (size_t) ldab * n;

    std::vector< scalar_t > AB( size_AB );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB.size(), &AB[0] );

    if (verbose >= 2) {
        printf( "AB = " ); print_matrix( kd+1, n, &AB[0], ldab );
    }

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = get_wtime();
    real_t norm_tst = lapack::lanhb( norm, uplo, n, kd, &AB[0], ldab );
    time = get_wtime() - time;

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::lanhb( norm, n, kd );
    //params.gflops() = gflop / time;

    if (verbose >= 1) {
        printf( "norm_tst = %.8e\n", norm_tst );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = get_wtime();
        real_t norm_ref = LAPACKE_lanhb( norm2char(norm), uplo2char(uplo), n, kd, &AB[0], ldab );
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
void test_lanhb( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_lanhb_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_lanhb_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_lanhb_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_lanhb_work< std::complex<double> >( params, run );
            break;
    }
}
