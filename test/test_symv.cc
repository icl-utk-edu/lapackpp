#include "test.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"
#include "check_gemm2.hh"

// some of this is copied from blaspp/test/cblas.hh
#ifdef HAVE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>

    // Original cblas.h used CBLAS_ORDER; new uses CBLAS_LAYOUT and makes
    // CBLAS_ORDER a typedef. Make sure CBLAS_LAYOUT is defined.
    typedef CBLAS_ORDER CBLAS_LAYOUT;
#endif

// -----------------------------------------------------------------------------
inline CBLAS_LAYOUT cblas_layout_const( blas::Layout layout )
{
    switch (layout) {
        case blas::Layout::RowMajor:  return CblasRowMajor;
        case blas::Layout::ColMajor:  return CblasColMajor;
        default: assert( false );
    }
}

inline CBLAS_UPLO cblas_uplo_const( blas::Uplo uplo )
{
    switch (uplo) {
        case blas::Uplo::Lower: return CblasLower;
        case blas::Uplo::Upper: return CblasUpper;
        default: assert( false );
    }
}

inline char lapack_uplo_const( CBLAS_UPLO uplo )
{
    switch (uplo) {
        case CblasLower: return 'l';
        case CblasUpper: return 'u';
        default:
            printf( "%s( %c )\n", __func__, uplo );
            assert( false );
    }
}

// -----------------------------------------------------------------------------
// give Fortran prototypes if not given via lapacke.h
#include "lapack_config.h"
#include "lapack_mangling.h"

extern "C" {

/* ----- symmetric rank-1 update */
#ifndef LAPACK_csymv
#define LAPACK_csymv LAPACK_GLOBAL(csymv,CSYMV)
void LAPACK_csymv(
        char const* uplo,
        lapack_int const* n,
        lapack_complex_float const* alpha,
        lapack_complex_float* A,
        lapack_int const* lda,
        lapack_complex_float const* x,
        lapack_int const* incx,
        lapack_complex_float* beta,
        lapack_complex_float const* yref,
        lapack_int const* incy );
#endif

#ifndef LAPACK_zsymv
#define LAPACK_zsymv LAPACK_GLOBAL(zsymv,ZSYMV)
void LAPACK_zsymv(
        char const* uplo,
        lapack_int const* n,
        lapack_complex_double const* alpha,
        lapack_complex_double* A,
        lapack_int const* lda,
        lapack_complex_double const* x,
        lapack_int const* incx,
        lapack_complex_double* beta,
        lapack_complex_double const* yref,
        lapack_int const* incy );
#endif

}  // extern "C"

inline void
lapack_symv(
        CBLAS_LAYOUT layout,
        CBLAS_UPLO uplo,
        lapack_int n,
        float alpha,
        float const* A,          lapack_int lda,
        float const* x,    lapack_int incx,
        float beta,
        float* yref, lapack_int incy )
{
    cblas_ssymv( layout, uplo, n, alpha, A, lda, x, incx, beta, yref, incy );
}

inline void
lapack_symv(
        CBLAS_LAYOUT layout,
        CBLAS_UPLO uplo,
        lapack_int n,
        double alpha,
        double const* A,          lapack_int lda,
        double const* x,    lapack_int incx,
        double beta,
        double* yref, lapack_int incy )
{
    cblas_dsymv( layout, uplo, n, alpha, A, lda, x, incx, beta, yref, incy );
}

inline void
lapack_symv(
        CBLAS_LAYOUT layout,
        CBLAS_UPLO uplo,
        lapack_int n,
        std::complex<float> alpha,
        std::complex<float> const* A,    lapack_int lda,
        std::complex<float> const* x,    lapack_int incx,
        std::complex<float> beta,
        std::complex<float>* yref, lapack_int incy
        )
{
    lapack_int n_ = n;
    lapack_int incx_ = incx;
    lapack_int incy_ = incy;
    lapack_int lda_ = lda;
    char uplo_ = lapack_uplo_const( uplo );
    if (layout == CblasRowMajor) {
        uplo_ = (uplo == CblasUpper ? 'l' : 'u');  // switch upper <=> lower
    }
    LAPACK_csymv( &uplo_, &n_,
                  (lapack_complex_float*) &alpha,
                  (lapack_complex_float*) A,
                  &lda_, x, &incx_, &beta, yref, &incy_ );
}

inline void
lapack_symv(
        CBLAS_LAYOUT layout,
        CBLAS_UPLO uplo,
        lapack_int n,
        std::complex<double> alpha,
        std::complex<double> const* A,          lapack_int lda,
        std::complex<double> const* x,    lapack_int incx,
        std::complex<double> beta,
        std::complex<double>* yref, lapack_int incy
)
{
    lapack_int n_ = n;
    lapack_int incx_ = incx;
    lapack_int incy_ = incy;
    lapack_int lda_ = lda;
    char uplo_ = lapack_uplo_const( uplo );
    if (layout == CblasRowMajor) {
        uplo_ = (uplo == CblasUpper ? 'l' : 'u');  // switch upper <=> lower
    }
    LAPACK_zsymv( &uplo_, &n_,
                  (lapack_complex_double*) &alpha,
                  (lapack_complex_double*) A,
                  &lda_, x, &incx_, &beta, yref, &incy_ );
}

// -----------------------------------------------------------------------------
template< typename TA, typename TX, typename TY >
void test_symv_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using blas::real;
    using blas::imag;
    using scalar_t = blas::scalar_type<TA, TX, TY>;
    using real_t = blas::real_type<scalar_t>;
    typedef long long lld;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Uplo uplo = params.uplo();
    scalar_t alpha  = params.alpha();
    scalar_t beta   = params.beta();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
    int64_t align   = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.gflops();
    params.gbytes();
    params.ref_time();
    params.ref_gflops();
    params.ref_gbytes();

    // adjust header to msec
    params.time.name( "BLAS++\ntime (ms)" );
    params.ref_time.name( "Ref.\ntime (ms)" );

    if (! run)
        return;

    // setup
    int64_t lda = roundup( n, align );
    size_t size_A = size_t(lda)*n;
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    size_t size_y = (n - 1) * std::abs(incy) + 1;
    std::vector<TA> A   ( size_A );
    std::vector<TX> x   ( size_x );
    std::vector<TY> y   ( size_y );
    std::vector<TY> yref( size_y );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::generate_matrix( params.matrix, n, n, &A[0], lda );
    lapack::larnv( idist, iseed, x.size(), &x[0] );
    lapack::larnv( idist, iseed, y.size(), &y[0] );
    yref = y;

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack::lansy( lapack::Norm::Fro, uplo, n, &A[0], lda );
    real_t Xnorm = blas::nrm2( n, &x[0], std::abs(incx) );
    real_t Ynorm = blas::nrm2( n, &y[0], std::abs(incy) );

    // test error exits
    if (params.error_exit() == 'y') {
        using blas::Layout;
        using blas::Uplo;
        assert_throw( blas::symv( Layout(0), uplo,     n, alpha, &A[0], lda, &x[0], incx, beta, &y[0], incy ), blas::Error );
        assert_throw( blas::symv( layout,    Uplo(0),  n, alpha, &A[0], lda, &x[0], incx, beta, &y[0], incy ), blas::Error );
        assert_throw( blas::symv( layout,    uplo,    -1, alpha, &A[0], lda, &x[0], incx, beta, &y[0], incy ), blas::Error );
        assert_throw( blas::symv( layout,    uplo,     n, alpha, &A[0], n-1, &x[0], incx, beta, &y[0], incy ), blas::Error );
        assert_throw( blas::symv( layout,    uplo,     n, alpha, &A[0], lda, &x[0],    0, beta, &y[0], incy ), blas::Error );
        assert_throw( blas::symv( layout,    uplo,     n, alpha, &A[0], lda, &x[0], incx, beta, &y[0],    0 ), blas::Error );
    }

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n"
                "y n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n",
                (lld) n, (lld) lda,  (lld) size_A, Anorm,
                (lld) n, (lld) incx, (lld) size_x, Xnorm,
                (lld) n, (lld) incy, (lld) size_y, Ynorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( n, n, &A[0], lda );
        printf( "x    = " ); print_vector( n, &x[0], incx );
        printf( "y    = " ); print_vector( n, &y[0], incy );
    }

    // run test
    libtest::flush_cache( params.cache() );
    double time = get_wtime();
    blas::symv( layout, uplo, n, alpha, &A[0], lda, &x[0], incx, beta, &y[0], incy );
    time = get_wtime() - time;

    double gflop = Gflop < scalar_t >::symv( n );
    double gbyte = Gbyte < scalar_t >::symv( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    if (verbose >= 2) {
        printf( "y2   = " ); print_vector( n, &y[0], incy );
    }

    if (params.check() == 'y') {
        // run reference
        libtest::flush_cache( params.cache() );
        time = get_wtime();
        lapack_symv( cblas_layout_const(layout), cblas_uplo_const(uplo), n,
                    alpha, &A[0], lda, &x[0], incx, beta, &yref[0], incy );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 2) {
            printf( "yref = " ); print_vector( n, &yref[0], incy );
        }

        // check error compared to reference
        // treat y as 1 x leny matrix with ld = incy; k = lenx is reduction dimension
        real_t error;
        int64_t okay;
        check_gemm( 1, n, n,
                alpha, beta,
                Anorm, Xnorm, Ynorm,
                &yref[0], std::abs(incy),
                &y[0], std::abs(incy),
                &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }
}

// -----------------------------------------------------------------------------
void test_symv( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();

        case libtest::DataType::Single:
            test_symv_work< float, float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_symv_work< double, double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_symv_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_symv_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;
    }
}