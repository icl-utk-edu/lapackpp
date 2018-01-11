#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_larft(
    char direct, char storev, lapack_int n, lapack_int k, float* V, lapack_int ldv, float* tau, float* T, lapack_int ldt )
{
    return LAPACKE_slarft( LAPACK_COL_MAJOR, direct, storev, n, k, V, ldv, tau, T, ldt );
}

static lapack_int LAPACKE_larft(
    char direct, char storev, lapack_int n, lapack_int k, double* V, lapack_int ldv, double* tau, double* T, lapack_int ldt )
{
    return LAPACKE_dlarft( LAPACK_COL_MAJOR, direct, storev, n, k, V, ldv, tau, T, ldt );
}

static lapack_int LAPACKE_larft(
    char direct, char storev, lapack_int n, lapack_int k, std::complex<float>* V, lapack_int ldv, std::complex<float>* tau, std::complex<float>* T, lapack_int ldt )
{
    return LAPACKE_clarft( LAPACK_COL_MAJOR, direct, storev, n, k, V, ldv, tau, T, ldt );
}

static lapack_int LAPACKE_larft(
    char direct, char storev, lapack_int n, lapack_int k, std::complex<double>* V, lapack_int ldv, std::complex<double>* tau, std::complex<double>* T, lapack_int ldt )
{
    return LAPACKE_zlarft( LAPACK_COL_MAJOR, direct, storev, n, k, V, ldv, tau, T, ldt );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_larft_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Direct direct = params.direct.value();
    lapack::StoreV storev = params.storev.value();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    int64_t align = params.align.value();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t ldv;
    if (storev == lapack::StoreV::Columnwise)
        ldv = roundup( max( 1, n ), align );
    else
        ldv = roundup( max( 1, k ), align );

    int64_t ldt = roundup( k, align );

    int lcv;
    size_t size_V;
    if (storev == lapack::StoreV::Columnwise) {
        lcv = k;
        size_V = (size_t) ldv * k;
    }
    else {
        lcv = n;
        size_V = (size_t) ldv * n;
    }

    size_t size_tau = (size_t) (k);
    size_t size_T = (size_t) ldt * k;

    std::vector< scalar_t > V( size_V );
    std::vector< scalar_t > tau( size_tau );
    std::vector< scalar_t > T_tst( size_T );
    std::vector< scalar_t > T_ref( size_T );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, V.size(), &V[0] );
    lapack::larnv( idist, iseed, T_tst.size(), &T_tst[0] );
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
        printf( "V = " ); print_matrix( ldv, lcv, &V[0], ldv );
        printf( "tau = " ); print_vector( k, &tau[0], 1 );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    lapack::larft( direct, storev, n, k, &V[0], ldv, &tau[0], &T_tst[0], ldt );
    time = get_wtime() - time;

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::larft( direct, storev, n, k );
    //params.gflops.value() = gflop / time;

    if (verbose >= 3) {
        printf( "T = " ); print_matrix( k, k, &T_tst[0], ldt );
    }

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_larft( direct2char(direct), storev2char(storev), n, k, &V[0], ldv, &tau[0], &T_ref[0], ldt );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_larft returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        if (verbose >= 3) {
            printf( "Tref = " ); print_matrix( k, k, &T_ref[0], ldt );
        }

        // ---------- check error compared to reference
        real_t error = 0;
        error += abs_error( T_tst, T_ref );
        params.error.value() = error;
        real_t tol = 100;
        real_t eps = std::numeric_limits< real_t >::epsilon();
        params.okay.value() = (error < tol*eps);  // todo: what's a good error check?
    }
}

// -----------------------------------------------------------------------------
void test_larft( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_larft_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_larft_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_larft_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_larft_work< std::complex<double> >( params, run );
            break;
    }
}
