#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>

#if LAPACK_VERSION >= 30600  // >= 3.6.0

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gesvdx(
    char jobu, char jobvt, char range, lapack_int m, lapack_int n, float* A, lapack_int lda, float vl, float vu, lapack_int il, lapack_int iu, lapack_int* ns, float* S, float* U, lapack_int ldu, float* VT, lapack_int ldvt )
{
    int64_t minmn = ( m < n ? m : n );
    std::vector< lapack_int > superb( 12 * minmn );
    return LAPACKE_sgesvdx( LAPACK_COL_MAJOR, jobu, jobvt, range, m, n, A, lda, vl, vu, il, iu, ns, S, U, ldu, VT, ldvt, &superb[0] );
}

static lapack_int LAPACKE_gesvdx(
    char jobu, char jobvt, char range, lapack_int m, lapack_int n, double* A, lapack_int lda, double vl, double vu, lapack_int il, lapack_int iu, lapack_int* ns, double* S, double* U, lapack_int ldu, double* VT, lapack_int ldvt )
{
    int64_t minmn = ( m < n ? m : n );
    std::vector< lapack_int > superb( 12 * minmn );
    return LAPACKE_dgesvdx( LAPACK_COL_MAJOR, jobu, jobvt, range, m, n, A, lda, vl, vu, il, iu, ns, S, U, ldu, VT, ldvt, &superb[0] );
}

static lapack_int LAPACKE_gesvdx(
    char jobu, char jobvt, char range, lapack_int m, lapack_int n, std::complex<float>* A, lapack_int lda, float vl, float vu, lapack_int il, lapack_int iu, lapack_int* ns, float* S, std::complex<float>* U, lapack_int ldu, std::complex<float>* VT, lapack_int ldvt )
{
    int64_t minmn = ( m < n ? m : n );
    std::vector< lapack_int > superb( 12 * minmn );
    return LAPACKE_cgesvdx( LAPACK_COL_MAJOR, jobu, jobvt, range, m, n, A, lda, vl, vu, il, iu, ns, S, U, ldu, VT, ldvt, &superb[0] );
}

static lapack_int LAPACKE_gesvdx(
    char jobu, char jobvt, char range, lapack_int m, lapack_int n, std::complex<double>* A, lapack_int lda, double vl, double vu, lapack_int il, lapack_int iu, lapack_int* ns, double* S, std::complex<double>* U, lapack_int ldu, std::complex<double>* VT, lapack_int ldvt )
{
    int64_t minmn = ( m < n ? m : n );
    std::vector< lapack_int > superb( 12 * minmn );
    return LAPACKE_zgesvdx( LAPACK_COL_MAJOR, jobu, jobvt, range, m, n, A, lda, vl, vu, il, iu, ns, S, U, ldu, VT, ldvt, &superb[0] );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gesvdx_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Job jobu = params.jobz.value();
    lapack::Job jobvt = params.jobvr.value();
    int64_t m = params.dim.m();
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

    if ( ( range==lapack::Range::Index ) &&
         ! ( ( 1 <= il ) && ( il < iu ) && ( iu < min( m, n ) ) ) ) {
        printf( "skipping because gesvdx requires 1 <= il <= iu <= min(m,n)\n" );
        return;
    }

    // ---------- setup
    int64_t lda = roundup( max( 1, m ), align );
    int64_t ns_tst;
    lapack_int ns_ref;
    int64_t ldu = ( jobu==lapack::Job::Vec ? roundup( m, align ) : 1 );
    int64_t ldvt = ( jobvt==lapack::Job::Vec ? roundup( min( m, n ), align ) : 1 );
    size_t size_A = (size_t) ( lda * n );
    size_t size_S = (size_t) ( min( m, n) );
    size_t size_U = (size_t) ( ldu * min( m, n ) );
    size_t size_VT = (size_t) ( ldvt * n );

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< real_t > S_tst( size_S );
    std::vector< real_t > S_ref( size_S );
    std::vector< scalar_t > U_tst( size_U );
    std::vector< scalar_t > U_ref( size_U );
    std::vector< scalar_t > VT_tst( size_VT );
    std::vector< scalar_t > VT_ref( size_VT );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    A_ref = A_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::gesvdx( jobu, jobvt, range, m, n, &A_tst[0], lda, vl, vu, il, iu, &ns_tst, &S_tst[0], &U_tst[0], ldu, &VT_tst[0], ldvt );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gesvdx returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::gesvdx( jobu, jobvt, range, m, n );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_gesvdx( job2char(jobu), job2char(jobvt), range2char(range), m, n, &A_ref[0], lda, vl, vu, il, iu, &ns_ref, &S_ref[0], &U_ref[0], ldu, &VT_ref[0], ldvt );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gesvdx returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += std::abs( ns_tst - ns_ref );
        error += abs_error( S_tst, S_ref );
        error += abs_error( U_tst, U_ref );
        error += abs_error( VT_tst, VT_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gesvdx( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gesvdx_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gesvdx_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gesvdx_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gesvdx_work< std::complex<double> >( params, run );
            break;
    }
}

#else

// -----------------------------------------------------------------------------
void test_gesvdx( Params& params, bool run )
{
    fprintf( stderr, "gesvdx requires LAPACK >= 3.6.0\n\n" );
    exit(0);
}

#endif  // LAPACK >= 3.6.0
