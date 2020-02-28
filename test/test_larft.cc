#include "test.hh"
#include "lapack.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_larft_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Direct direct = params.direct();
    lapack::StoreV storev = params.storev();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    int64_t align = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();

    if (! run)
        return;

    // skip invalid sizes
    if (! (n >= k)) {
        params.msg() = "skipping: requires n >= k (not documented)";
        return;
    }

    // ---------- setup
    int64_t Vm, Vn;
    if (storev == lapack::StoreV::Columnwise) {
        Vm = n;
        Vn = k;
    }
    else {
        Vm = k;
        Vn = n;
    }
    int64_t ldv = roundup( blas::max( 1, Vm ), align );
    int64_t ldt = roundup( blas::max( 1, k ), align );
    size_t size_V   = (size_t) ldv * Vn;
    size_t size_tau = (size_t) (k);
    size_t size_T   = (size_t) ldt * k;

    std::vector< scalar_t > V( size_V );
    std::vector< scalar_t > tau( size_tau );
    std::vector< scalar_t > T_tst( size_T );
    std::vector< scalar_t > T_ref( size_T );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, V.size(), &V[0] );
    lapack::generate_matrix( params.matrix, k, k, &T_tst[0], ldt );
    T_ref = T_tst;

    // generate Householder vectors; initializes tau
    // From larft docs, with n = 5 and k = 3:
    // direct = 'f' and storev = 'c':         direct = 'f' and storev = 'r':
    //
    //              V = (  1       )                 V = (  1 v1 v1 v1 v1 )
    //                  ( v1  1    )                     (     1 v2 v2 v2 )
    //                  ( v1 v2  1 )                     (        1 v3 v3 )
    //                  ( v1 v2 v3 )
    //                  ( v1 v2 v3 )
    //
    // direct = 'b' and storev = 'c':         direct = 'b' and storev = 'r':
    //
    //              V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
    //                  ( v1 v2 v3 )                     ( v2 v2 v2  1    )
    //                  (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
    //                  (     1 v3 )
    //                  (        1 )
    for (int i = 0; i < k; ++i) {
        if (storev == lapack::StoreV::Columnwise) {
            if (direct == lapack::Direct::Forward) {
                lapack::larfg( n-i, &V[i + i*ldv], &V[i+1 + i*ldv], 1, &tau[i] );
            }
            else {
                lapack::larfg( n-k+i+1, &V[(n - k + i) + i*ldv], &V[0 + i*ldv], 1, &tau[i] );
            }
        }
        else {
            if (direct == lapack::Direct::Forward) {
                lapack::larfg( n-i, &V[i + i*ldv], &V[i + (i+1)*ldv], ldv, &tau[i] );
            }
            else {
                lapack::larfg( n-k+i+1, &V[i + (n - k + i)*ldv], &V[i + 0*ldv], ldv, &tau[i] );
            }
        }
    }

    if (verbose >= 2) {
        printf( "V = " ); print_matrix( Vm, Vn, &V[0], ldv );
        printf( "tau = " ); print_vector( k, &tau[0], 1 );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    lapack::larft( direct, storev, n, k, &V[0], ldv, &tau[0], &T_tst[0], ldt );
    time = testsweeper::get_wtime() - time;

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::larft( direct, storev, n, k );
    //params.gflops() = gflop / time;

    if (verbose >= 3) {
        printf( "T = " ); print_matrix( k, k, &T_tst[0], ldt );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_larft( direct2char(direct), storev2char(storev), n, k, &V[0], ldv, &tau[0], &T_ref[0], ldt );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_larft returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;

        if (verbose >= 3) {
            printf( "Tref = " ); print_matrix( k, k, &T_ref[0], ldt );
        }

        // ---------- check error compared to reference
        real_t error = 0;
        error += abs_error( T_tst, T_ref );
        params.error() = error;
        real_t tol = 100;
        real_t eps = std::numeric_limits< real_t >::epsilon();
        params.okay() = (error < tol*eps);  // todo: what's a good error check?
    }
}

// -----------------------------------------------------------------------------
void test_larft( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_larft_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_larft_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_larft_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_larft_work< std::complex<double> >( params, run );
            break;
    }
}
