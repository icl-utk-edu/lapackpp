#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

#include "lapack_mangling.h"
extern "C" {
// give Fortran prototypes if not given via lapacke.h

#ifndef LAPACK_slanst
#define LAPACK_slanst LAPACK_GLOBAL(slanst,SLANST)
blas_float_return LAPACK_slanst(
    char const* norm, lapack_int const* n,
    float const* D,
    float const* E );
#endif

#ifndef LAPACK_dlanst
#define LAPACK_dlanst LAPACK_GLOBAL(dlanst,DLANST)
double LAPACK_dlanst(
    char const* norm, lapack_int const* n,
    double const* D,
    double const* E );
#endif

#ifndef LAPACK_clanht
#define LAPACK_clanht LAPACK_GLOBAL(clanht,CLANHT)
blas_float_return LAPACK_clanht(
    char const* norm, lapack_int const* n,
    float const* D,
    lapack_complex_float const* E );
#endif

#ifndef LAPACK_zlanht
#define LAPACK_zlanht LAPACK_GLOBAL(zlanht,ZLANHT)
double LAPACK_zlanht(
    char const* norm, lapack_int const* n,
    double const* D,
    lapack_complex_double const* E );
#endif

}  // extern "C"

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACK (not in LAPACKE)
static lapack_int LAPACKE_lanht(
    char norm, lapack_int n, float* D, float* E )
{
    return LAPACK_slanst( &norm, &n, D, E );
}

static lapack_int LAPACKE_lanht(
    char norm, lapack_int n, double* D, double* E )
{
    return LAPACK_dlanst( &norm, &n, D, E );
}

static lapack_int LAPACKE_lanht(
    char norm, lapack_int n, float* D, std::complex<float>* E )
{
    return LAPACK_clanht( &norm, &n, D, E );
}

static lapack_int LAPACKE_lanht(
    char norm, lapack_int n, double* D, std::complex<double>* E )
{
    return LAPACK_zlanht( &norm, &n, D, E );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_lanht_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Norm norm = params.norm.value();
    int64_t n = params.dim.n();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    size_t size_D = (size_t) (n);
    size_t size_E = (size_t) (n-1);

    std::vector< real_t > D( size_D );
    std::vector< scalar_t > E( size_E );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, D.size(), &D[0] );
    lapack::larnv( idist, iseed, E.size(), &E[0] );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t norm_tst = lapack::lanht( norm, n, &D[0], &E[0] );
    time = get_wtime() - time;

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::lanht( norm, n );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t norm_ref = LAPACKE_lanht( norm2char(norm), n, &D[0], &E[0] );
        time = get_wtime() - time;

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (norm_tst != norm_ref) {
            error = 1;
        }
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_lanht( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_lanht_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_lanht_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_lanht_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_lanht_work< std::complex<double> >( params, run );
            break;
    }
}
