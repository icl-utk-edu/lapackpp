#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

#include "lapack_mangling.h"
extern "C" {
// give Fortran prototypes if not given via lapacke.h

#ifndef LAPACK_slangt
#define LAPACK_slangt LAPACK_GLOBAL(slangt,SLANGT)
blas_float_return LAPACK_slangt(
    char const* norm, lapack_int const* n,
    float const* DL,
    float const* D,
    float const* DU );
#endif

#ifndef LAPACK_dlangt
#define LAPACK_dlangt LAPACK_GLOBAL(dlangt,DLANGT)
double LAPACK_dlangt(
    char const* norm, lapack_int const* n,
    double const* DL,
    double const* D,
    double const* DU );
#endif

#ifndef LAPACK_clangt
#define LAPACK_clangt LAPACK_GLOBAL(clangt,CLANGT)
blas_float_return LAPACK_clangt(
    char const* norm, lapack_int const* n,
    lapack_complex_float const* DL,
    lapack_complex_float const* D,
    lapack_complex_float const* DU );
#endif

#ifndef LAPACK_zlangt
#define LAPACK_zlangt LAPACK_GLOBAL(zlangt,ZLANGT)
double LAPACK_zlangt(
    char const* norm, lapack_int const* n,
    lapack_complex_double const* DL,
    lapack_complex_double const* D,
    lapack_complex_double const* DU );
#endif

}  // extern "C"

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACK (not in LAPACKE)
static lapack_int LAPACKE_langt(
    char norm, lapack_int n, float* DL, float* D, float* DU )
{
    return LAPACK_slangt( &norm, &n, DL, D, DU );
}

static lapack_int LAPACKE_langt(
    char norm, lapack_int n, double* DL, double* D, double* DU )
{
    return LAPACK_dlangt( &norm, &n, DL, D, DU );
}

static lapack_int LAPACKE_langt(
    char norm, lapack_int n, std::complex<float>* DL, std::complex<float>* D, std::complex<float>* DU )
{
    return LAPACK_clangt( &norm, &n, DL, D, DU );
}

static lapack_int LAPACKE_langt(
    char norm, lapack_int n, std::complex<double>* DL, std::complex<double>* D, std::complex<double>* DU )
{
    return LAPACK_zlangt( &norm, &n, DL, D, DU );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_langt_work( Params& params, bool run )
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
    size_t size_DL = (size_t) (n-1);
    size_t size_D = (size_t) (n);
    size_t size_DU = (size_t) (n-1);

    std::vector< scalar_t > DL( size_DL );
    std::vector< scalar_t > D( size_D );
    std::vector< scalar_t > DU( size_DU );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, DL.size(), &DL[0] );
    lapack::larnv( idist, iseed, D.size(), &D[0] );
    lapack::larnv( idist, iseed, DU.size(), &DU[0] );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t norm_tst = lapack::langt( norm, n, &DL[0], &D[0], &DU[0] );
    time = get_wtime() - time;

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::langt( norm, n );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t norm_ref = LAPACKE_langt( norm2char(norm), n, &DL[0], &D[0], &DU[0] );
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
void test_langt( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_langt_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_langt_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_langt_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_langt_work< std::complex<double> >( params, run );
            break;
    }
}
