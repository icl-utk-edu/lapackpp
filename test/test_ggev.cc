#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"

#include <vector>
#include <iostream>

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_ggev(
    char jobvl, char jobvr, lapack_int n, float* A, lapack_int lda, float* B, lapack_int ldb, std::complex<float>* alpha, float* beta, float* VL, lapack_int ldvl, float* VR, lapack_int ldvr )
{
    std::vector< float > alphar( n ), alphai( n );
    lapack_int err = LAPACKE_sggev( LAPACK_COL_MAJOR, jobvl, jobvr, n, A, lda, B, ldb, &alphar[0], &alphai[0], beta, VL, ldvl, VR, ldvr );
    for (int64_t i = 0; i < n; ++i) {
        alpha[i] = std::complex<float>( alphar[i], alphai[i] );
    }
    return err;
}

static lapack_int LAPACKE_ggev(
    char jobvl, char jobvr, lapack_int n, double* A, lapack_int lda, double* B, lapack_int ldb, std::complex<double>* alpha, double* beta, double* VL, lapack_int ldvl, double* VR, lapack_int ldvr )
{
    std::vector< double > alphar( n ), alphai( n );
    lapack_int err = LAPACKE_dggev( LAPACK_COL_MAJOR, jobvl, jobvr, n, A, lda, B, ldb, &alphar[0], &alphai[0], beta, VL, ldvl, VR, ldvr );
    for (int64_t i = 0; i < n; ++i) {
        alpha[i] = std::complex<double>( alphar[i], alphai[i] );
    }
    return err;
}

static lapack_int LAPACKE_ggev(
    char jobvl, char jobvr, lapack_int n, std::complex<float>* A, lapack_int lda, std::complex<float>* B, lapack_int ldb, std::complex<float>* alpha, std::complex<float>* beta, std::complex<float>* VL, lapack_int ldvl, std::complex<float>* VR, lapack_int ldvr )
{
    return LAPACKE_cggev( LAPACK_COL_MAJOR, jobvl, jobvr, n, A, lda, B, ldb, alpha, beta, VL, ldvl, VR, ldvr );
}

static lapack_int LAPACKE_ggev(
    char jobvl, char jobvr, lapack_int n, std::complex<double>* A, lapack_int lda, std::complex<double>* B, lapack_int ldb, std::complex<double>* alpha, std::complex<double>* beta, std::complex<double>* VL, lapack_int ldvl, std::complex<double>* VR, lapack_int ldvr )
{
    return LAPACKE_zggev( LAPACK_COL_MAJOR, jobvl, jobvr, n, A, lda, B, ldb, alpha, beta, VL, ldvl, VR, ldvr );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_ggev_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    lapack::Job jobvl = params.jobvl.value();
    lapack::Job jobvr = params.jobvr.value();
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
    int64_t ldb = roundup( max( 1, n ), align );
    int64_t ldvl = ( jobvl==lapack::Job::Vec ? roundup( max(1, n), align ) : 1 );
    int64_t ldvr  = ( jobvr==lapack::Job::Vec ? roundup( max(1, n), align ) : 1 );
    size_t size_A = (size_t)( lda * n );
    size_t size_B = (size_t)( ldb * n );
    size_t size_alpha = (size_t)( n );
    size_t size_beta = (size_t)( n );
    size_t size_VL = (size_t)( ldvl * n );
    size_t size_VR = (size_t)( ldvr * n );

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );
    std::vector< std::complex<real_t> > alpha_tst( size_alpha );
    std::vector< std::complex<real_t> > alpha_ref( size_alpha );
    std::vector< scalar_t > beta_tst( size_beta );
    std::vector< scalar_t > beta_ref( size_beta );
    std::vector< scalar_t > VL_tst( size_VL );
    std::vector< scalar_t > VL_ref( size_VL );
    std::vector< scalar_t > VR_tst( size_VR );
    std::vector< scalar_t > VR_ref( size_VR );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, A_tst.size(), &A_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );
    A_ref = A_tst;
    B_ref = B_tst;

    std::copy( alpha_tst.begin(), alpha_tst.end(), alpha_ref.begin() );

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::ggev( jobvl, jobvr, n, &A_tst[0], lda, &B_tst[0], ldb, &alpha_tst[0], &beta_tst[0], &VL_tst[0], ldvl, &VR_tst[0], ldvr );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::ggev returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::ggev( jobvl, jobvr, n );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_ggev( job2char(jobvl), job2char(jobvr), n, &A_ref[0], lda, &B_ref[0], ldb, &alpha_ref[0], &beta_ref[0], &VL_ref[0], ldvl, &VR_ref[0], ldvr );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_ggev returned error %lld\n", (lld) info_ref );
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
        error += abs_error( alpha_tst, alpha_ref );
        error += abs_error( beta_tst, beta_ref );
        error += abs_error( VL_tst, VL_ref );
        error += abs_error( VR_tst, VR_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_ggev( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_ggev_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_ggev_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_ggev_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_ggev_work< std::complex<double> >( params, run );
            break;
    }
}
