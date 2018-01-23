#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

#include "lapack_mangling.h"
extern "C" {
// give Fortran prototypes if not given via lapacke.h
#ifndef LAPACK_slantp
#define LAPACK_slantp LAPACK_GLOBAL(slantp,SLANTP)
blas_float_return LAPACK_slantp(
    char const* norm, char const* uplo, char const* diag,
    lapack_int const* n,
    float const* AP,
    float* work );
#endif

#ifndef LAPACK_dlantp
#define LAPACK_dlantp LAPACK_GLOBAL(dlantp,DLANTP)
double LAPACK_dlantp(
    char const* norm, char const* uplo, char const* diag,
    lapack_int const* n,
    double const* AP,
    double* work );
#endif

#ifndef LAPACK_clantp
#define LAPACK_clantp LAPACK_GLOBAL(clantp,CLANTP)
blas_float_return LAPACK_clantp(
    char const* norm, char const* uplo, char const* diag,
    lapack_int const* n,
    lapack_complex_float const* AP,
    float* work );
#endif

#ifndef LAPACK_zlantp
#define LAPACK_zlantp LAPACK_GLOBAL(zlantp,ZLANTP)
double LAPACK_zlantp(
    char const* norm, char const* uplo, char const* diag,
    lapack_int const* n,
    lapack_complex_double const* AP,
    double* work );
#endif
}  // extern "C"

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACK (not in LAPACKE)
static lapack_int LAPACKE_lantp(
    char norm, char uplo, char diag, lapack_int n, float* AP )
{
    std::vector< float > work( n );
    return LAPACK_slantp( &norm, &uplo, &diag, &n, AP, &work[0] );
}

static lapack_int LAPACKE_lantp(
    char norm, char uplo, char diag, lapack_int n, double* AP )
{
    std::vector< double > work( n );
    return LAPACK_dlantp( &norm, &uplo, &diag, &n, AP, &work[0] );
}

static lapack_int LAPACKE_lantp(
    char norm, char uplo, char diag, lapack_int n, std::complex<float>* AP )
{
    std::vector< float > work( n );
    return LAPACK_clantp( &norm, &uplo, &diag, &n, AP, &work[0] );
}

static lapack_int LAPACKE_lantp(
    char norm, char uplo, char diag, lapack_int n, std::complex<double>* AP )
{
    std::vector< double > work( n );
    return LAPACK_zlantp( &norm, &uplo, &diag, &n, AP, &work[0] );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_lantp_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Norm norm = params.norm.value();
    lapack::Uplo uplo = params.uplo.value();
    lapack::Diag diag = params.diag.value();
    int64_t n = params.dim.n();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    size_t size_AP = (size_t) (n*(n+1)/2);

    std::vector< scalar_t > AP( size_AP );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AP.size(), &AP[0] );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t norm_tst = lapack::lantp( norm, uplo, diag, n, &AP[0] );
    time = get_wtime() - time;

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::lantp( norm, diag, n );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t norm_ref = LAPACKE_lantp( norm2char(norm), uplo2char(uplo), diag2char(diag), n, &AP[0] );
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
void test_lantp( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_lantp_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_lantp_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_lantp_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_lantp_work< std::complex<double> >( params, run );
            break;
    }
}
