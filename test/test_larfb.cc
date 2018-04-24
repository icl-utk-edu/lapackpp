#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_larfb_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Side side = params.side.value();
    lapack::Op trans = params.trans.value();
    lapack::Direct direct = params.direct.value();
    lapack::StoreV storev = params.storev.value();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    if ((side == lapack::Side::Left  && m < k) ||
        (side == lapack::Side::Right && n < k))
    {
        printf( "skipping because larfb requires m >= k >= 0 (left) or n >= k >= 0 (right)\n" );
        return;
    }

    // ---------- setup
    int64_t ldv;
    if (storev == lapack::StoreV::Columnwise) {
        if (side == lapack::Side::Left)
            ldv = roundup( max( 1, m ), align );
        else
            ldv = roundup( max( 1, n ), align );
    }
    else {
        // rowwise
        ldv = roundup( k, align );
    }

    int64_t ldt = roundup( k, align );
    int64_t ldc = roundup( max( 1, m ), align );

    size_t size_V;
    if (storev == lapack::StoreV::Columnwise) {
        size_V = (size_t) ldv * k;
    }
    else {
        // rowwise
        if (side == lapack::Side::Left)
            size_V = (size_t) ldv * m;
        else if (side == lapack::Side::Right)
            size_V = (size_t) ldv * n;
    }

    size_t size_T = (size_t) ldt * k;
    size_t size_C = (size_t) ldc * n;

    std::vector< scalar_t > V( size_V );
    std::vector< scalar_t > T( size_T );
    std::vector< scalar_t > C_tst( size_C );
    std::vector< scalar_t > C_ref( size_C );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, V.size(), &V[0] );
    lapack::larnv( idist, iseed, T.size(), &T[0] );
    lapack::larnv( idist, iseed, C_tst.size(), &C_tst[0] );
    C_ref = C_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    lapack::larfb( side, trans, direct, storev, m, n, k, &V[0], ldv, &T[0], ldt, &C_tst[0], ldc );
    time = get_wtime() - time;

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::larfb( side, trans, direct, storev, m, n, k );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_larfb( side2char(side), op2char(trans), direct2char(direct), storev2char(storev), m, n, k, &V[0], ldv, &T[0], ldt, &C_ref[0], ldc );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_larfb returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        error += abs_error( C_tst, C_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_larfb( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_larfb_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_larfb_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_larfb_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_larfb_work< std::complex<double> >( params, run );
            break;
    }
}
