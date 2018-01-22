#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

#if LAPACK_VERSION_MAJOR >= 3 && LAPACK_VERSION_MINOR >= 7  // >= 3.7

// TODO Remove Fortran prototypes when LAPACKE applies the bugfix
#include "lapack_mangling.h"

extern "C" {

/* ----- symmetric indefinite factorization, Aasen's */
#ifndef LAPACK_ssytrf_aa
#define LAPACK_ssytrf_aa LAPACK_GLOBAL(ssytrf_aa,SSYTRF_AA)
void LAPACK_ssytrf_aa(
    char const* uplo,
    lapack_int const* n,
    float* a, lapack_int const* lda,
    lapack_int* ipiv,
    float* work, lapack_int const* lwork,
    lapack_int* info );
#endif

#ifndef LAPACK_dsytrf_aa
#define LAPACK_dsytrf_aa LAPACK_GLOBAL(dsytrf_aa,DSYTRF_AA)
void LAPACK_dsytrf_aa(
    char const* uplo,
    lapack_int const* n,
    double* a, lapack_int const* lda,
    lapack_int* ipiv,
    double* work, lapack_int const* lwork,
    lapack_int* info );
#endif

#ifndef LAPACK_csytrf_aa
#define LAPACK_csytrf_aa LAPACK_GLOBAL(csytrf_aa,CSYTRF_AA)
void LAPACK_csytrf_aa(
    char const* uplo,
    lapack_int const* n,
    lapack_complex_float* a, lapack_int const* lda,
    lapack_int* ipiv,
    lapack_complex_float* work, lapack_int const* lwork,
    lapack_int* info );
#endif

#ifndef LAPACK_zsytrf_aa
#define LAPACK_zsytrf_aa LAPACK_GLOBAL(zsytrf_aa,ZSYTRF_AA)
void LAPACK_zsytrf_aa(
    char const* uplo,
    lapack_int const* n,
    lapack_complex_double* a, lapack_int const* lda,
    lapack_int* ipiv,
    lapack_complex_double* work, lapack_int const* lwork,
    lapack_int* info );
#endif

#ifndef LAPACK_chetrf_aa
#define LAPACK_chetrf_aa LAPACK_GLOBAL(chetrf_aa,CHETRF_AA)
void LAPACK_chetrf_aa(
    char const* uplo,
    lapack_int const* n,
    lapack_complex_float* a, lapack_int const* lda,
    lapack_int* ipiv,
    lapack_complex_float* work, lapack_int const* lwork,
    lapack_int* info );
#endif

#ifndef LAPACK_zhetrf_aa
#define LAPACK_zhetrf_aa LAPACK_GLOBAL(zhetrf_aa,ZHETRF_AA)
void LAPACK_zhetrf_aa(
    char const* uplo,
    lapack_int const* n,
    lapack_complex_double* a, lapack_int const* lda,
    lapack_int* ipiv,
    lapack_complex_double* work, lapack_int const* lwork,
    lapack_int* info );
#endif

}  // extern "C"

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_sytrf_aa(
    char uplo, lapack_int n, float* A, lapack_int lda, lapack_int* ipiv )
{
//  TODO Uncomment this when LAPACKE applies the bugfix
    // return LAPACKE_ssytrf_aa( LAPACK_COL_MAJOR, uplo, n, A, lda, ipiv );
    blas_int info_ = 0;
    blas_int lwork_ = n*128;
    std::vector< float > work( lwork_ );
    LAPACK_ssytrf_aa( &uplo, &n, A, &lda, ipiv, &work[0], &lwork_, &info_ );
    return info_;
}

static lapack_int LAPACKE_sytrf_aa(
    char uplo, lapack_int n, double* A, lapack_int lda, lapack_int* ipiv )
{
//  TODO Uncomment this when LAPACKE applies the bugfix
    // return LAPACKE_dsytrf_aa( LAPACK_COL_MAJOR, uplo, n, A, lda, ipiv );
    blas_int info_ = 0;
    blas_int lwork_ = n*128;
    std::vector< double > work( lwork_ );
    LAPACK_dsytrf_aa( &uplo, &n, A, &lda, ipiv, &work[0], &lwork_, &info_ );
    return info_;
}

static lapack_int LAPACKE_sytrf_aa(
    char uplo, lapack_int n, std::complex<float>* A, lapack_int lda, lapack_int* ipiv )
{
//  TODO Uncomment this when LAPACKE applies the bugfix
    // return LAPACKE_csytrf_aa( LAPACK_COL_MAJOR, uplo, n, A, lda, ipiv );
    blas_int info_ = 0;
    blas_int lwork_ = n*128;
    std::vector< std::complex<float> > work( lwork_ );
    LAPACK_csytrf_aa( &uplo, &n, A, &lda, ipiv, &work[0], &lwork_, &info_ );
    return info_;
}

static lapack_int LAPACKE_sytrf_aa(
    char uplo, lapack_int n, std::complex<double>* A, lapack_int lda, lapack_int* ipiv )
{
//  TODO Uncomment this when LAPACKE applies the bugfix
    // return LAPACKE_zsytrf_aa( LAPACK_COL_MAJOR, uplo, n, A, lda, ipiv );
    blas_int info_ = 0;
    blas_int lwork_ = n*128;
    std::vector< std::complex<double> > work( lwork_ );
    LAPACK_zsytrf_aa( &uplo, &n, A, &lda, ipiv, &work[0], &lwork_, &info_ );
    return info_;
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_sytrf_aa_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_ipiv = (size_t) (n);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    A_ref = A_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    int64_t info_tst = lapack::sytrf_aa( uplo, n, &A_tst[0], lda, &ipiv_tst[0] );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::sytrf_aa returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::sytrf_aa( n );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_sytrf_aa( uplo2char(uplo), n, &A_ref[0], lda, &ipiv_ref[0] );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_sytrf_aa returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( ipiv_tst, ipiv_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_sytrf_aa( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_sytrf_aa_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_sytrf_aa_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_sytrf_aa_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_sytrf_aa_work< std::complex<double> >( params, run );
            break;
    }
}

#else

// -----------------------------------------------------------------------------
void test_sytrf_aa( Params& params, bool run )
{
    fprintf( stderr, "sytrf_aa requires LAPACK >= 3.7\n\n" );
    exit(0);
}

#endif  // LAPACK >= 3.7

