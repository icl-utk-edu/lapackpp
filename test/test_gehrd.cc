#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include "check_gehrd.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gehrd(
    lapack_int n, lapack_int ilo, lapack_int ihi, float* A, lapack_int lda, float* tau )
{
    return LAPACKE_sgehrd( LAPACK_COL_MAJOR, n, ilo, ihi, A, lda, tau );
}

static lapack_int LAPACKE_gehrd(
    lapack_int n, lapack_int ilo, lapack_int ihi, double* A, lapack_int lda, double* tau )
{
    return LAPACKE_dgehrd( LAPACK_COL_MAJOR, n, ilo, ihi, A, lda, tau );
}

static lapack_int LAPACKE_gehrd(
    lapack_int n, lapack_int ilo, lapack_int ihi, std::complex<float>* A, lapack_int lda, std::complex<float>* tau )
{
    return LAPACKE_cgehrd( LAPACK_COL_MAJOR, n, ilo, ihi, A, lda, tau );
}

static lapack_int LAPACKE_gehrd(
    lapack_int n, lapack_int ilo, lapack_int ihi, std::complex<double>* A, lapack_int lda, std::complex<double>* tau )
{
    return LAPACKE_zgehrd( LAPACK_COL_MAJOR, n, ilo, ihi, A, lda, tau );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gehrd_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t ilo = 1;
    int64_t ihi = n;
    int64_t align = params.align.value();
    int64_t verbose = params.verbose.value();
    params.matrix.mark();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol.value() * eps;

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();
    params.gflops.value();
    params.ortho.value();

    params.error.name( "A - U H U^H\nerror" );
    params.ortho.name( "I - U U^H\nerror" );

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_tau = (size_t) (n-1);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > tau_tst( size_tau );
    std::vector< scalar_t > tau_ref( size_tau );

    lapack::generate_matrix( params.matrix, n, n, nullptr, &A_tst[0], lda );
    A_ref = A_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::gehrd( n, ilo, ihi, &A_tst[0], lda, &tau_tst[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gehrd returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    double gflop = lapack::Gflop< scalar_t >::gehrd( n );
    params.gflops.value() = gflop / time;

    if (params.check.value() == 'y') {
        // ---------- check numerical error
        real_t results[2];
        check_gehrd( n, &A_ref[0], lda, &A_tst[0], lda, &tau_tst[0],
                     verbose, results );
        params.error.value() = results[0];
        params.ortho.value() = results[1];
        params.okay.value() = (results[0] < tol && results[1] < tol);
    }

    if (params.ref.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_gehrd( n, ilo, ihi, &A_ref[0], lda, &tau_ref[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gehrd returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        params.ref_gflops.value() = gflop / time;
    }
}

// -----------------------------------------------------------------------------
void test_gehrd( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gehrd_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gehrd_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gehrd_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gehrd_work< std::complex<double> >( params, run );
            break;
    }
}
