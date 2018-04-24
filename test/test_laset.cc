#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_laset_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::MatrixType matrixtype = params.matrixtype.value();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    scalar_t alpha = params.alpha.value();
    scalar_t beta = params.beta.value();
    int64_t align = params.align.value();
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, m ), align );
    size_t size_A = (size_t) lda * n;

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );

    lapack::generate_matrix( params.matrix, m, n, nullptr, &A_tst[0], lda );
    A_ref = A_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    lapack::laset( matrixtype, m, n, alpha, beta, &A_tst[0], lda );
    time = get_wtime() - time;

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::laset( m, n, alpha, beta );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_laset( matrixtype2char(matrixtype), m, n, alpha, beta, &A_ref[0], lda );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_laset returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        error += abs_error( A_tst, A_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_laset( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_laset_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_laset_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_laset_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_laset_work< std::complex<double> >( params, run );
            break;
    }
}
