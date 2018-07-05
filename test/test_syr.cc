// this is similar to blaspp/test/test_syr.hh,
// except it uses LAPACK++ instead of calling Fortran LAPACK,
// and tests syr< complex >.


#include "test.hh"
#include "cblas.hh"
#include "lapack.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"
#include "check_gemm2.hh"  // uses lapack++ instead of Fortran lapack

#include "syr.hh"  // from blaspp

// -----------------------------------------------------------------------------
// give Fortran prototypes if not given via lapacke.h
#include "lapack_config.h"
#include "lapack_mangling.h"

extern "C" {

/* ----- symmetric rank-1 update */
#ifndef LAPACK_csyr
#define LAPACK_csyr LAPACK_GLOBAL(csyr,CSYR)
void LAPACK_csyr(
    char const* uplo,
    lapack_int const* n,
    lapack_complex_float const* alpha,
    lapack_complex_float const* x, lapack_int const* incx,
    lapack_complex_float* a, lapack_int const* lda );
#endif

#ifndef LAPACK_zsyr
#define LAPACK_zsyr LAPACK_GLOBAL(zsyr,ZSYR)
void LAPACK_zsyr(
    char const* uplo,
    lapack_int const* n,
    lapack_complex_double const* alpha,
    lapack_complex_double const* x, lapack_int const* incx,
    lapack_complex_double* a, lapack_int const* lda );
#endif

}  // extern "C"

// -----------------------------------------------------------------------------
inline void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float>* A, int lda )
{
    char uplo_ = lapack_uplo_const( uplo );
    if (layout == CblasRowMajor) {
        uplo_ = (uplo == CblasUpper ? 'l' : 'u');  // switch upper <=> lower
    }
    LAPACK_csyr( &uplo_, &n,
                 (lapack_complex_float*) &alpha,
                 (lapack_complex_float*) x, &incx,
                 (lapack_complex_float*) A, &lda );
}

inline void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double>* A, int lda )
{
    char uplo_ = lapack_uplo_const( uplo );
    if (layout == CblasRowMajor) {
        uplo_ = (uplo == CblasUpper ? 'l' : 'u');  // switch upper <=> lower
    }
    LAPACK_zsyr( &uplo_, &n,
                 (lapack_complex_double*) &alpha,
                 (lapack_complex_double*) x, &incx,
                 (lapack_complex_double*) A, &lda );
}

// -----------------------------------------------------------------------------
template< typename TA, typename TX >
void test_syr_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using scalar_t = blas::scalar_type< TA, TX >;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    blas::Layout layout = params.layout.value();
    blas::Uplo uplo = params.uplo.value();
    scalar_t alpha  = params.alpha.value();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx.value();
    int64_t align   = params.align.value();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();
    params.gflops.value();

    if (! run)
        return;

    // setup
    int64_t lda = roundup( n, align );
    size_t size_A = size_t(lda)*n;
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    TA* A    = new TA[ size_A ];
    TA* Aref = new TA[ size_A ];
    TX* x    = new TX[ size_x ];

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 0, 0, 1 };
    lapack::larnv( idist, iseed, size_x, x );
    lapack::generate_matrix( params.matrix, n, n, &A[0], lda );
    Aref = A;

    // norms for error check
    real_t Anorm = lapack::lansy( lapack::Norm::Fro, uplo, n, A, lda );
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );

    // test error exits
    assert_throw( blas::syr( Layout(0), uplo,     n, alpha, x, incx, A, lda ), blas::Error );
    assert_throw( blas::syr( layout,    Uplo(0),  n, alpha, x, incx, A, lda ), blas::Error );
    assert_throw( blas::syr( layout,    uplo,    -1, alpha, x, incx, A, lda ), blas::Error );
    assert_throw( blas::syr( layout,    uplo,     n, alpha, x,    0, A, lda ), blas::Error );
    assert_throw( blas::syr( layout,    uplo,     n, alpha, x, incx, A, n-1 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n",
                (lld) n, (lld) lda,  (lld) size_A, Anorm,
                (lld) n, (lld) incx, (lld) size_x, Xnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei;\n",
                real(alpha), imag(alpha) );
        printf( "A = " ); print_matrix( n, n, A, lda );
        printf( "x = " ); print_vector( n, x, incx );
    }

    // run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    blas::syr( layout, uplo, n, alpha, x, incx, A, lda );
    time = get_wtime() - time;

    params.time.value() = time * 1000;  // msec
    double gflop = Gflop< scalar_t >::syr( n );
    params.gflops.value() = gflop / time;

    if (verbose >= 2) {
        printf( "A2 = " ); print_matrix( n, n, A, lda );
    }

    if (params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        cblas_syr( cblas_layout_const(layout), cblas_uplo_const(uplo),
                   n, alpha, x, incx, Aref, lda );
        time = get_wtime() - time;

        params.ref_time.value() = time * 1000;  // msec
        params.ref_gflops.value() = gflop / time;

        if (verbose >= 2) {
            printf( "Aref = " ); print_matrix( n, n, Aref, lda );
        }

        // check error compared to reference
        // beta = 1
        real_t error;
        int64_t okay;
        check_herk( uplo, n, 1, alpha, scalar_t(1), Xnorm, Xnorm, Anorm,
                    Aref, lda, A, lda, &error, &okay );
        params.error.value() = error;
        params.okay.value() = okay;
    }
}

// -----------------------------------------------------------------------------
void test_syr( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_syr_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_syr_work< float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_syr_work< double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_syr_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_syr_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;
    }
}
