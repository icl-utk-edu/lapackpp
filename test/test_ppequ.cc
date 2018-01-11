#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_ppequ(
    char uplo, lapack_int n, float* AP, float* S, float* scond, float* amax )
{
    return LAPACKE_sppequ( LAPACK_COL_MAJOR, uplo, n, AP, S, scond, amax );
}

static lapack_int LAPACKE_ppequ(
    char uplo, lapack_int n, double* AP, double* S, double* scond, double* amax )
{
    return LAPACKE_dppequ( LAPACK_COL_MAJOR, uplo, n, AP, S, scond, amax );
}

static lapack_int LAPACKE_ppequ(
    char uplo, lapack_int n, std::complex<float>* AP, float* S, float* scond, float* amax )
{
    return LAPACKE_cppequ( LAPACK_COL_MAJOR, uplo, n, AP, S, scond, amax );
}

static lapack_int LAPACKE_ppequ(
    char uplo, lapack_int n, std::complex<double>* AP, double* S, double* scond, double* amax )
{
    return LAPACKE_zppequ( LAPACK_COL_MAJOR, uplo, n, AP, S, scond, amax );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_ppequ_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    real_t scond_tst = 0;
    real_t scond_ref = 0;
    real_t amax_tst;
    real_t amax_ref;
    size_t size_AP = (size_t) (n*(n+1)/2);
    size_t size_S = (size_t) (n);

    std::vector< scalar_t > AP( size_AP );
    std::vector< real_t > S_tst( size_S );
    std::vector< real_t > S_ref( size_S );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AP.size(), &AP[0] );

    // diagonally dominant -> positive definite
    if (uplo == lapack::Uplo::Upper) {
        for (int64_t i = 0; i < n; ++i) {
            AP[ i + 0.5*(i+1)*i ] += n;
        }
    }
    else { // lower
        for (int64_t i = 0; i < n; ++i) {
            AP[ i + n*i - 0.5*i*(i+1) ] += n;
        }
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::ppequ( uplo, n, &AP[0], &S_tst[0], &scond_tst, &amax_tst );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::ppequ returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::ppequ( n );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_ppequ( uplo2char(uplo), n, &AP[0], &S_ref[0], &scond_ref, &amax_ref );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_ppequ returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( S_tst, S_ref );
        error += std::abs( scond_tst - scond_ref );
        error += std::abs( amax_tst - amax_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_ppequ( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_ppequ_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_ppequ_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_ppequ_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_ppequ_work< std::complex<double> >( params, run );
            break;
    }
}
