#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_pbequ(
    char uplo, lapack_int n, lapack_int kd, float* AB, lapack_int ldab, float* S, float* scond, float* amax )
{
    return LAPACKE_spbequ( LAPACK_COL_MAJOR, uplo, n, kd, AB, ldab, S, scond, amax );
}

static lapack_int LAPACKE_pbequ(
    char uplo, lapack_int n, lapack_int kd, double* AB, lapack_int ldab, double* S, double* scond, double* amax )
{
    return LAPACKE_dpbequ( LAPACK_COL_MAJOR, uplo, n, kd, AB, ldab, S, scond, amax );
}

static lapack_int LAPACKE_pbequ(
    char uplo, lapack_int n, lapack_int kd, std::complex<float>* AB, lapack_int ldab, float* S, float* scond, float* amax )
{
    return LAPACKE_cpbequ( LAPACK_COL_MAJOR, uplo, n, kd, AB, ldab, S, scond, amax );
}

static lapack_int LAPACKE_pbequ(
    char uplo, lapack_int n, lapack_int kd, std::complex<double>* AB, lapack_int ldab, double* S, double* scond, double* amax )
{
    return LAPACKE_zpbequ( LAPACK_COL_MAJOR, uplo, n, kd, AB, ldab, S, scond, amax );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_pbequ_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    int64_t kd = params.kd.value();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( kd+1, align );
    real_t scond_tst = 0;
    real_t scond_ref = 0;
    real_t amax_tst;
    real_t amax_ref;
    size_t size_AB = (size_t) ldab * n;
    size_t size_S = (size_t) (n);

    std::vector< scalar_t > AB( size_AB );
    std::vector< real_t > S_tst( size_S );
    std::vector< real_t > S_ref( size_S );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB.size(), &AB[0] );

    // diagonally dominant -> positive definite
    if (uplo == lapack::Uplo::Upper) {
        for (int64_t j = 0; j < n; ++j) {
            AB[ kd + j*ldab ] += n;
        }
    }
    else { // lower
        for (int64_t j = 0; j < n; ++j) {
            AB[ j*ldab ] += n;
        }
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::pbequ( uplo, n, kd, &AB[0], ldab, &S_tst[0], &scond_tst, &amax_tst );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::pbequ returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::pbequ( n, kd );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_pbequ( uplo2char(uplo), n, kd, &AB[0], ldab, &S_ref[0], &scond_ref, &amax_ref );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_pbequ returned error %lld\n", (lld) info_ref );
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
void test_pbequ( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_pbequ_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_pbequ_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_pbequ_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_pbequ_work< std::complex<double> >( params, run );
            break;
    }
}
