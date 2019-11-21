#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_upmtr_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Side side = params.side();
    lapack::Uplo uplo = params.uplo();
    lapack::Op trans = params.trans();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align();

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t ldc = roundup( blas::max( 1, m ), align );
    int64_t r = ( side == lapack::Side::Left ) ? m : n;
    size_t size_AP = (size_t) (r*(r+1)/2);
    size_t size_tau = (size_t) blas::max( 1, r-1 );
    size_t size_C = (size_t) ldc * n;
    size_t size_D = (size_t) (r);
    size_t size_E = (size_t) (r-1);

    std::vector< scalar_t > AP( size_AP );
    std::vector< scalar_t > tau( size_tau );
    std::vector< scalar_t > C_tst( size_C );
    std::vector< scalar_t > C_ref( size_C );
    std::vector< real_t > D( size_D );
    std::vector< real_t > E( size_E );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AP.size(), &AP[0] );
    lapack::larnv( idist, iseed, tau.size(), &tau[0] );
    lapack::larnv( idist, iseed, C_tst.size(), &C_tst[0] );
    C_ref = C_tst;

    int64_t info = lapack::hptrd( uplo, r, &AP[0], &D[0], &E[0], &tau[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::hptrd returned error %lld\n", (lld) info );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::upmtr( side, uplo, trans, m, n, &AP[0], &tau[0], &C_tst[0], ldc );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::upmtr returned error %lld\n", (lld) info_tst );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::upmtr( side, trans, m, n );
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_upmtr( side2char(side), uplo2char(uplo), op2char(trans), m, n, &AP[0], &tau[0], &C_ref[0], ldc );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_upmtr returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( C_tst, C_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_upmtr( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_upmtr_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_upmtr_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_upmtr_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_upmtr_work< std::complex<double> >( params, run );
            break;
    }
}
