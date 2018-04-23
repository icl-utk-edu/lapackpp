#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_heevr(
    char jobz, char range, char uplo, lapack_int n, float* A, lapack_int lda, float vl, float vu, lapack_int il, lapack_int iu, float abstol, lapack_int* m, float* W, float* Z, lapack_int ldz, lapack_int* isuppz )
{
    return LAPACKE_ssyevr( LAPACK_COL_MAJOR, jobz, range, uplo, n, A, lda, vl, vu, il, iu, abstol, m, W, Z, ldz, isuppz );
}

static lapack_int LAPACKE_heevr(
    char jobz, char range, char uplo, lapack_int n, double* A, lapack_int lda, double vl, double vu, lapack_int il, lapack_int iu, double abstol, lapack_int* m, double* W, double* Z, lapack_int ldz, lapack_int* isuppz )
{
    return LAPACKE_dsyevr( LAPACK_COL_MAJOR, jobz, range, uplo, n, A, lda, vl, vu, il, iu, abstol, m, W, Z, ldz, isuppz );
}

static lapack_int LAPACKE_heevr(
    char jobz, char range, char uplo, lapack_int n, std::complex<float>* A, lapack_int lda, float vl, float vu, lapack_int il, lapack_int iu, float abstol, lapack_int* m, float* W, std::complex<float>* Z, lapack_int ldz, lapack_int* isuppz )
{
    return LAPACKE_cheevr( LAPACK_COL_MAJOR, jobz, range, uplo, n, A, lda, vl, vu, il, iu, abstol, m, W, Z, ldz, isuppz );
}

static lapack_int LAPACKE_heevr(
    char jobz, char range, char uplo, lapack_int n, std::complex<double>* A, lapack_int lda, double vl, double vu, lapack_int il, lapack_int iu, double abstol, lapack_int* m, double* W, std::complex<double>* Z, lapack_int ldz, lapack_int* isuppz )
{
    return LAPACKE_zheevr( LAPACK_COL_MAJOR, jobz, range, uplo, n, A, lda, vl, vu, il, iu, abstol, m, W, Z, ldz, isuppz );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_heevr_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Job jobz = params.jobz.value();
    lapack::Uplo uplo = params.uplo.value();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();

    real_t  vl;  // = params.vl.value();
    real_t  vu;  // = params.vu.value();
    int64_t il;  // = params.il.value();
    int64_t iu;  // = params.iu.value();
    lapack::Range range;  // derived from vl,vu,il,iu
    params.get_range( n, &range, &vl, &vu, &il, &iu );
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    real_t abstol = 0;  // default value
    int64_t m_tst;
    lapack_int m_ref;
    int64_t ldz = roundup( max( 1, n ), align );
    size_t size_A = (size_t) ( lda * n );
    size_t size_W = (size_t) ( n );
    size_t size_Z = (size_t) ( ldz * max( 1, n ) );
    size_t size_isuppz = (size_t) ( 2 * max( 1, n ) );

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< real_t > W_tst( size_W );
    std::vector< real_t > W_ref( size_W );
    std::vector< scalar_t > Z_tst( size_Z );
    std::vector< scalar_t > Z_ref( size_Z );
    std::vector< int64_t > isuppz_tst( size_isuppz );
    std::vector< lapack_int > isuppz_ref( size_isuppz );

    lapack::generate_matrix( params.matrix, n, n, nullptr, &A_tst[0], lda );
    A_ref = A_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::heevr( jobz, range, uplo, n, &A_tst[0], lda, vl, vu, il, iu, abstol, &m_tst, &W_tst[0], &Z_tst[0], ldz, &isuppz_tst[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::heevr returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::heevr( jobz, range, n );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_heevr( job2char(jobz), range2char(range), uplo2char(uplo), n, &A_ref[0], lda, vl, vu, il, iu, abstol, &m_ref, &W_ref[0], &Z_ref[0], ldz, &isuppz_ref[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_heevr returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += std::abs( m_tst - m_ref );
        error += abs_error( W_tst, W_ref );
        error += abs_error( Z_tst, Z_ref );
        // Only check valid ifail values
        // ifail is referenced only if eigenvectors are needed (jobz =
        // 'V') and all eigenvalues are needed, that is, range = 'A'
        // or range = 'I' and il = 1 and iu = n.
        for ( size_t i = 0; i < (size_t)(2*m_ref); i++ ) 
            if ( ( jobz==lapack::Job::Vec ) &&
                 ( ( range==lapack::Range::All ) || 
                   ( range==lapack::Range::Index && il==1 && iu==n ) ) )  {
                error += std::abs( isuppz_tst[i] - isuppz_ref[i] );
            }
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_heevr( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_heevr_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_heevr_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_heevr_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_heevr_work< std::complex<double> >( params, run );
            break;
    }
}
