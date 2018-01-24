#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

#include "lapack_mangling.h"
extern "C" {
// give Fortran prototypes if not given via lapacke.h

#ifndef LAPACK_slantb
#define LAPACK_slantb LAPACK_GLOBAL(slantb,SLANTB)
blas_float_return LAPACK_slantb(
    char const* norm, char const* uplo, char const* diag,
    lapack_int const* n, lapack_int const* k,
    float const* AB, lapack_int const* ldab,
    float* work );
#endif

#ifndef LAPACK_dlantb
#define LAPACK_dlantb LAPACK_GLOBAL(dlantb,DLANTB)
double LAPACK_dlantb(
    char const* norm, char const* uplo, char const* diag,
    lapack_int const* n, lapack_int const* k,
    double const* AB, lapack_int const* ldab,
    double* work );
#endif

#ifndef LAPACK_clantb
#define LAPACK_clantb LAPACK_GLOBAL(clantb,CLANTB)
blas_float_return LAPACK_clantb(
    char const* norm, char const* uplo, char const* diag,
    lapack_int const* n, lapack_int const* k,
    lapack_complex_float const* AB, lapack_int const* ldab,
    float* work );
#endif

#ifndef LAPACK_zlantb
#define LAPACK_zlantb LAPACK_GLOBAL(zlantb,ZLANTB)
double LAPACK_zlantb(
    char const* norm, char const* uplo, char const* diag,
    lapack_int const* n, lapack_int const* k,
    lapack_complex_double const* AB, lapack_int const* ldab,
    double* work );
#endif

}  // extern "C"

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACK (not in LAPACKE)
static lapack_int LAPACKE_lantb(
    char norm, char uplo, char diag, lapack_int n, lapack_int k, float* AB, lapack_int ldab )
{
    std::vector< float > work( n );
    return LAPACK_slantb( &norm, &uplo, &diag, &n, &k, AB, &ldab, &work[0] );
}

static lapack_int LAPACKE_lantb(
    char norm, char uplo, char diag, lapack_int n, lapack_int k, double* AB, lapack_int ldab )
{
    std::vector< double > work( n );
    return LAPACK_dlantb( &norm, &uplo, &diag, &n, &k, AB, &ldab, &work[0] );
}

static lapack_int LAPACKE_lantb(
    char norm, char uplo, char diag, lapack_int n, lapack_int k, std::complex<float>* AB, lapack_int ldab )
{
    std::vector< float > work( n );
    return LAPACK_clantb( &norm, &uplo, &diag, &n, &k, AB, &ldab, &work[0] );
}

static lapack_int LAPACKE_lantb(
    char norm, char uplo, char diag, lapack_int n, lapack_int k, std::complex<double>* AB, lapack_int ldab )
{
    std::vector< double > work( n );
    return LAPACK_zlantb( &norm, &uplo, &diag, &n, &k, AB, &ldab, &work[0] );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_lantb_work( Params& params, bool run )
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
    int64_t k = params.kd.value();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( k+1, align );
    size_t size_AB = (size_t) ldab * n;

    std::vector< scalar_t > AB( size_AB );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB.size(), &AB[0] );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t norm_tst = lapack::lantb( norm, uplo, diag, n, k, &AB[0], ldab );
    time = get_wtime() - time;

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::lantb( norm, diag, n, k );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t norm_ref = LAPACKE_lantb( norm2char(norm), uplo2char(uplo), diag2char(diag), n, k, &AB[0], ldab );
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
void test_lantb( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_lantb_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_lantb_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_lantb_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_lantb_work< std::complex<double> >( params, run );
            break;
    }
}
