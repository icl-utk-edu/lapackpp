#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_langb_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Norm norm = params.norm.value();
    int64_t n = params.dim.n();
    int64_t kl = params.kl.value();
    int64_t ku = params.ku.value();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( kl+ku+1, align );
    size_t size_AB = (size_t) ldab * n;

    std::vector< scalar_t > AB( size_AB );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB.size(), &AB[0] );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t norm_tst = lapack::langb( norm, n, kl, ku, &AB[0], ldab );
    time = get_wtime() - time;

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::langb( norm, n, kl, ku );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t norm_ref = LAPACKE_langb( norm2char(norm), n, kl, ku, &AB[0], ldab );
        time = get_wtime() - time;

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (norm_tst != norm_ref) {
            error = 1;
        }
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_langb( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_langb_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_langb_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_langb_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_langb_work< std::complex<double> >( params, run );
            break;
    }
}
