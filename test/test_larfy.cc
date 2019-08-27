#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

#if LAPACK_VERSION >= 30700  // >= 3.7.0

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_larfy_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t incv = params.incx();
    int64_t align = params.align();

    // mark non-standard output values
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();

    if (! run)
        return;

    // ---------- setup
    scalar_t tau;  // todo value
    int64_t ldc = roundup( blas::max(  1, n  ), align );
    size_t size_V = (size_t) (1 + (n-1)*abs(incv));
    size_t size_C = (size_t) ldc * n;

    std::vector< scalar_t > V( size_V );
    std::vector< scalar_t > C_tst( size_C );
    std::vector< scalar_t > C_ref( size_C );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, V.size(), &V[0] );
    lapack::larnv( idist, iseed, 1, &tau );
    lapack::generate_matrix( params.matrix, n, n, &C_tst[0], ldc );
    C_ref = C_tst;

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = libtest::get_wtime();
    lapack::larfy( uplo, n, &V[0], incv, tau, &C_tst[0], ldc );
    time = libtest::get_wtime() - time;

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::larfy( n );
    //params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = libtest::get_wtime();
        int64_t info_ref = LAPACKE_larfy( uplo2char(uplo), n, &V[0], incv, tau, &C_ref[0], ldc );
        time = libtest::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_larfy returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        error += abs_error( C_tst, C_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_larfy( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_larfy_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_larfy_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_larfy_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_larfy_work< std::complex<double> >( params, run );
            break;
    }
}

#else

// -----------------------------------------------------------------------------
void test_larfy( Params& params, bool run )
{
    fprintf( stderr, "larfy requires LAPACK >= 3.7.0\n\n" );
    exit(0);
}

#endif  // LAPACK >= 3.7.0