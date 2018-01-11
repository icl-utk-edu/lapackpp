#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

#include "lapack_mangling.h"

extern "C" {

/* ----- matrix norm - symmetric packed */
// give Fortran prototypes if not given via lapacke.h
#ifndef LAPACK_slansp
#define LAPACK_slansp LAPACK_GLOBAL(slansp,SLANSP)
blas_float_return LAPACK_slansp(
    char const* norm, char const* uplo,
    lapack_int const* n,
    float const* AP, float* work );
#endif

#ifndef LAPACK_dlansp
#define LAPACK_dlansp LAPACK_GLOBAL(dlansp,DLANSP)
double LAPACK_dlansp(
    char const* norm, char const* uplo,
    lapack_int const* n,
    double const* AP, double* work );
#endif

#ifndef LAPACK_clansp
#define LAPACK_clansp LAPACK_GLOBAL(clansp,CLANSP)
blas_float_return LAPACK_clansp(
    char const* norm, char const* uplo,
    lapack_int const* n,
    lapack_complex_float const* AP, float* work );
#endif

#ifndef LAPACK_zlansp
#define LAPACK_zlansp LAPACK_GLOBAL(zlansp,ZLANSP)
double LAPACK_zlansp(
    char const* norm, char const* uplo,
    lapack_int const* n,
    lapack_complex_double const* AP, double* work );
#endif

}  // extern "C"

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACK (not in LAPACKE)
static float LAPACKE_lansp(
    char norm, char uplo,
    lapack_int n,
    float const* AP )
{
    std::vector< float > work( n );
    return LAPACK_slansp( &norm, &uplo, &n, AP, &work[0] );
}

static double LAPACKE_lansp(
    char norm, char uplo,
    lapack_int n,
    double const* AP )
{
    std::vector< double > work( n );
    return LAPACK_dlansp( &norm, &uplo, &n, AP, &work[0] );
}

static float LAPACKE_lansp(
    char norm, char uplo,
    lapack_int n,
    lapack_complex_float const* AP )
{
    std::vector< float > work( n );
    return LAPACK_clansp( &norm, &uplo, &n, AP, &work[0] );
}

static double LAPACKE_lansp(
    char norm, char uplo,
    lapack_int n,
    lapack_complex_double const* AP )
{
    std::vector< double > work( n );
    return LAPACK_zlansp( &norm, &uplo, &n, AP, &work[0] );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_lansp_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Norm norm = params.norm.value();
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    size_t size_AP = (size_t) (n*(n+1)/2);

    std::vector< scalar_t > AP( size_AP );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AP.size(), &AP[0] );

    if (verbose >= 2) {
        printf( "AP = " ); print_vector( AP.size(), &AP[0], 1 );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    real_t norm_tst = lapack::lansp( norm, uplo, n, &AP[0] );
    time = get_wtime() - time;

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::lansp( norm, n );
    //params.gflops.value() = gflop / time;

    if (verbose >= 1) {
        printf( "norm_tst = %.8e\n", norm_tst );
    }

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        real_t norm_ref = LAPACKE_lansp( norm2char(norm), uplo2char(uplo), n, &AP[0] );
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
void test_lansp( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_lansp_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_lansp_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_lansp_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_lansp_work< std::complex<double> >( params, run );
            break;
    }
}
