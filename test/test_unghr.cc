#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_unghr(
    lapack_int n, lapack_int ilo, lapack_int ihi, float* A, lapack_int lda, float* tau )
{
    return LAPACKE_sorghr( LAPACK_COL_MAJOR, n, ilo, ihi, A, lda, tau );
}

static lapack_int LAPACKE_unghr(
    lapack_int n, lapack_int ilo, lapack_int ihi, double* A, lapack_int lda, double* tau )
{
    return LAPACKE_dorghr( LAPACK_COL_MAJOR, n, ilo, ihi, A, lda, tau );
}

static lapack_int LAPACKE_unghr(
    lapack_int n, lapack_int ilo, lapack_int ihi, std::complex<float>* A, lapack_int lda, std::complex<float>* tau )
{
    return LAPACKE_cunghr( LAPACK_COL_MAJOR, n, ilo, ihi, A, lda, tau );
}

static lapack_int LAPACKE_unghr(
    lapack_int n, lapack_int ilo, lapack_int ihi, std::complex<double>* A, lapack_int lda, std::complex<double>* tau )
{
    return LAPACKE_zunghr( LAPACK_COL_MAJOR, n, ilo, ihi, A, lda, tau );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_unghr_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t ilo = 1; // TODO params.ilo.value();
    int64_t ihi = n; // TODO params.ihi.value();
    int64_t align = params.align.value();
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_tau = (size_t) (n-1);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > tau( size_tau );

    lapack::generate_matrix( params.matrix, n, n, nullptr, &A_tst[0], lda );

    // reduce A to Hessenberg form
    int64_t info_hrd = lapack::gehrd( n, ilo, ihi, &A_tst[0], lda, &tau[0] );
    if (info_hrd != 0) {
        fprintf( stderr, "lapack::gehrd returned error %lld\n", (lld) info_hrd );
    }

    A_ref = A_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::unghr( n, ilo, ihi, &A_tst[0], lda, &tau[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::unghr returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::unghr( n, ilo, ihi );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_unghr( n, ilo, ihi, &A_ref[0], lda, &tau[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_unghr returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_unghr( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_unghr_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_unghr_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_unghr_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_unghr_work< std::complex<double> >( params, run );
            break;
    }
}
