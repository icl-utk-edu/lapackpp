#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_lantb_work( Params& params, bool run )
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
    int64_t k = params.kd();
    int64_t align = params.align();

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( k+1, align );
    size_t size_AB = (size_t) ldab * n;

    std::vector< scalar_t > AB( size_AB );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB.size(), &AB[0] );

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = get_wtime();
    int64_t norm_tst = lapack::lantb( norm, uplo, diag, n, k, &AB[0], ldab );
    time = get_wtime() - time;

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::lantb( norm, diag, n, k );
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = get_wtime();
        int64_t norm_ref = LAPACKE_lantb( norm2char(norm), uplo2char(uplo), diag2char(diag), n, k, &AB[0], ldab );
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
void test_lantb( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_lantb_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_lantb_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_lantb_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_lantb_work< std::complex<double> >( params, run );
            break;
    }
}
