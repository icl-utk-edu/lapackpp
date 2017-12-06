#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <omp.h>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_gesdd(
    char jobz, lapack_int m, lapack_int n, float* A, lapack_int lda, float* S, float* U, lapack_int ldu, float* VT, lapack_int ldvt )
{
    return LAPACKE_sgesdd( LAPACK_COL_MAJOR, jobz, m, n, A, lda, S, U, ldu, VT, ldvt );
}

static lapack_int LAPACKE_gesdd(
    char jobz, lapack_int m, lapack_int n, double* A, lapack_int lda, double* S, double* U, lapack_int ldu, double* VT, lapack_int ldvt )
{
    return LAPACKE_dgesdd( LAPACK_COL_MAJOR, jobz, m, n, A, lda, S, U, ldu, VT, ldvt );
}

static lapack_int LAPACKE_gesdd(
    char jobz, lapack_int m, lapack_int n, std::complex<float>* A, lapack_int lda, float* S, std::complex<float>* U, lapack_int ldu, std::complex<float>* VT, lapack_int ldvt )
{
    return LAPACKE_cgesdd( LAPACK_COL_MAJOR, jobz, m, n, A, lda, S, U, ldu, VT, ldvt );
}

static lapack_int LAPACKE_gesdd(
    char jobz, lapack_int m, lapack_int n, std::complex<double>* A, lapack_int lda, double* S, std::complex<double>* U, lapack_int ldu, std::complex<double>* VT, lapack_int ldvt )
{
    return LAPACKE_zgesdd( LAPACK_COL_MAJOR, jobz, m, n, A, lda, S, U, ldu, VT, ldvt );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gesdd_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Job jobu = params.jobu.value();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t ucol = (jobu == lapack::Job::AllVec ? m : min( m, n ));
    int64_t lda = roundup( max( 1, m ), align );
    int64_t ldu = roundup( m, align );
    int64_t ldvt = roundup( (jobu == lapack::Job::AllVec ? n : min( m, n )), align );
    size_t size_A = (size_t) lda * n;
    size_t size_S = (size_t) (min(m,n));
    size_t size_U = (size_t) ldu * ucol;
    size_t size_VT = (size_t) ldvt * n;

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
    double time = omp_get_wtime();
    int64_t info_tst = lapack::gesdd( jobu, m, n, &A_tst[0], lda, &S_tst[0], &U_tst[0], ldu, &VT_tst[0], ldvt );
    time = omp_get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gesdd returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::gesdd( jobu, m, n );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        int64_t info_ref = LAPACKE_gesdd( job2char(jobu), m, n, &A_ref[0], lda, &S_ref[0], &U_ref[0], ldu, &VT_ref[0], ldvt );
        time = omp_get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gesdd returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( S_tst, S_ref );
        error += abs_error( U_tst, U_ref );
        error += abs_error( VT_tst, VT_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gesdd( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gesdd_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gesdd_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gesdd_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gesdd_work< std::complex<double> >( params, run );
            break;
    }
}
