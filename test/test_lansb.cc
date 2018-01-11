#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

#include "lapack_mangling.h"

extern "C" {

/* ----- matrix norm - symmetric banded */
// give Fortran prototypes if not given via lapacke.h
#ifndef LAPACK_slansb
#define LAPACK_slansb LAPACK_GLOBAL(slansb,SLANSB)
blas_float_return LAPACK_slansb(
    char const* norm, char const* uplo,
    lapack_int const* n, lapack_int const* kd,
    float const* AB, lapack_int const* ldab,
    float* work );
#endif

#ifndef LAPACK_dlansb
#define LAPACK_dlansb LAPACK_GLOBAL(dlansb,DLANSB)
double LAPACK_dlansb(
    char const* norm, char const* uplo,
    lapack_int const* n, lapack_int const* kd,
    double const* AB, lapack_int const* ldab,
    double* work );
#endif

#ifndef LAPACK_clansb
#define LAPACK_clansb LAPACK_GLOBAL(clansb,CLANSB)
blas_float_return LAPACK_clansb(
    char const* norm, char const* uplo,
    lapack_int const* n, lapack_int const* kd,
    lapack_complex_float const* AB, lapack_int const* ldab,
    float* work );
#endif

#ifndef LAPACK_zlansb
#define LAPACK_zlansb LAPACK_GLOBAL(zlansb,ZLANSB)
double LAPACK_zlansb(
    char const* norm, char const* uplo,
    lapack_int const* n, lapack_int const* kd,
    lapack_complex_double const* AB, lapack_int const* ldab,
    double* work );
#endif

}  // extern "C"

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACK (not in LAPACKE)
static float LAPACKE_lansb(
    char norm, char uplo, lapack_int n, lapack_int kd, float* AB, lapack_int ldab )
{
    std::vector< float > work( n );
    return LAPACK_slansb( &norm, &uplo, &n, &kd, AB, &ldab, &work[0] );
}

static double LAPACKE_lansb(
    char norm, char uplo, lapack_int n, lapack_int kd, double* AB, lapack_int ldab )
{
    std::vector< double > work( n );
    return LAPACK_dlansb( &norm, &uplo, &n, &kd, AB, &ldab, &work[0] );
}

static float LAPACKE_lansb(
    char norm, char uplo, lapack_int n, lapack_int kd, std::complex<float>* AB, lapack_int ldab )
{
    std::vector< float > work( n );
    return LAPACK_clansb( &norm, &uplo, &n, &kd, AB, &ldab, &work[0] );
}

static double LAPACKE_lansb(
    char norm, char uplo, lapack_int n, lapack_int kd, std::complex<double>* AB, lapack_int ldab )
{
    std::vector< double > work( n );
    return LAPACK_zlansb( &norm, &uplo, &n, &kd, AB, &ldab, &work[0] );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_lansb_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Norm norm = params.norm.value();
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    int64_t kd = min( params.kd.value(), n-1 );
    int64_t align = params.align.value();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( kd+1, align );
    size_t size_AB = (size_t) ldab * n;

    std::vector< scalar_t > AB( size_AB );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB.size(), &AB[0] );

    if (verbose >= 2) {
        printf( "AB = " ); print_matrix( kd+1, n, &AB[0], ldab );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    real_t norm_tst = lapack::lansb( norm, uplo, n, kd, &AB[0], ldab );
    time = get_wtime() - time;

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::lansb( norm, n, kd );
    //params.gflops.value() = gflop / time;

    if (verbose >= 1) {
        printf( "norm_tst = %.8e\n", norm_tst );
    }

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        real_t norm_ref = LAPACKE_lansb( norm2char(norm), uplo2char(uplo), n, kd, &AB[0], ldab );
        time = get_wtime() - time;

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        if (verbose >= 1) {
            printf( "norm_ref = %.8e\n", norm_ref );
        }

        // ---------- check error compared to reference
        real_t error = 0;
        error += std::abs( norm_tst - norm_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_lansb( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_lansb_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_lansb_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_lansb_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_lansb_work< std::complex<double> >( params, run );
            break;
    }
}
