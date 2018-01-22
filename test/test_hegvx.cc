#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_hegvx(
    lapack_int itype, char jobz, char range, char uplo, lapack_int n, float* A, lapack_int lda, float* B, lapack_int ldb, float vl, float vu, lapack_int il, lapack_int iu, float abstol, lapack_int* m, float* W, float* Z, lapack_int ldz, lapack_int* ifail )
{
    return LAPACKE_ssygvx( LAPACK_COL_MAJOR, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, m, W, Z, ldz, ifail );
}

static lapack_int LAPACKE_hegvx(
    lapack_int itype, char jobz, char range, char uplo, lapack_int n, double* A, lapack_int lda, double* B, lapack_int ldb, double vl, double vu, lapack_int il, lapack_int iu, double abstol, lapack_int* m, double* W, double* Z, lapack_int ldz, lapack_int* ifail )
{
    return LAPACKE_dsygvx( LAPACK_COL_MAJOR, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, m, W, Z, ldz, ifail );
}

static lapack_int LAPACKE_hegvx(
    lapack_int itype, char jobz, char range, char uplo, lapack_int n, std::complex<float>* A, lapack_int lda, std::complex<float>* B, lapack_int ldb, float vl, float vu, lapack_int il, lapack_int iu, float abstol, lapack_int* m, float* W, std::complex<float>* Z, lapack_int ldz, lapack_int* ifail )
{
    return LAPACKE_chegvx( LAPACK_COL_MAJOR, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, m, W, Z, ldz, ifail );
}

static lapack_int LAPACKE_hegvx(
    lapack_int itype, char jobz, char range, char uplo, lapack_int n, std::complex<double>* A, lapack_int lda, std::complex<double>* B, lapack_int ldb, double vl, double vu, lapack_int il, lapack_int iu, double abstol, lapack_int* m, double* W, std::complex<double>* Z, lapack_int ldz, lapack_int* ifail )
{
    return LAPACKE_zhegvx( LAPACK_COL_MAJOR, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, abstol, m, W, Z, ldz, ifail );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_hegvx_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    int64_t itype = params.itype.value();
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

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    int64_t ldb = roundup( max( 1, n ), align );
    real_t abstol = 0;  // default value
    int64_t m_tst;
    lapack_int m_ref;
    int64_t ldz = ( jobz==lapack::Job::NoVec ? 1 : roundup( max( 1, n ), align ) );
    size_t size_A = (size_t) ( lda * n );
    size_t size_B = (size_t) ( ldb * n );
    size_t size_W = (size_t) ( n );
    size_t size_Z = (size_t) max( 1, ldz * n );
    size_t size_ifail = (size_t) ( n );

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );
    std::vector< real_t > W_tst( size_W );
    std::vector< real_t > W_ref( size_W );
    std::vector< scalar_t > Z_tst( size_Z );
    std::vector< scalar_t > Z_ref( size_Z );
    std::vector< int64_t > ifail_tst( size_ifail );
    std::vector< lapack_int > ifail_ref( size_ifail );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    // diagonally dominant -> positive definite
    for (int64_t i = 0; i < n; ++i) {
        B_tst[ i + i*ldb ] += n;
    }
    A_ref = A_tst;
    B_ref = B_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::hegvx( itype, jobz, range, uplo, n, &A_tst[0], lda, &B_tst[0], ldb, vl, vu, il, iu, abstol, &m_tst, &W_tst[0], &Z_tst[0], ldz, &ifail_tst[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::hegvx returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::hegvx( itype, jobz, range, n );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_hegvx( itype, job2char(jobz), range2char(range), uplo2char(uplo), n, &A_ref[0], lda, &B_ref[0], ldb, vl, vu, il, iu, abstol, &m_ref, &W_ref[0], &Z_ref[0], ldz, &ifail_ref[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_hegvx returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( B_tst, B_ref );
        error += std::abs( m_tst - m_ref );
        error += abs_error( W_tst, W_ref );
        error += abs_error( Z_tst, Z_ref );
        // Check first m elements of ifail
        if ( jobz==lapack::Job::Vec ) {
            for ( size_t i = 0; i < (size_t)(m_ref); i++ ) 
                error += std::abs( ifail_tst[i] - ifail_ref[i] );
        }
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_hegvx( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_hegvx_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_hegvx_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_hegvx_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_hegvx_work< std::complex<double> >( params, run );
            break;
    }
}
