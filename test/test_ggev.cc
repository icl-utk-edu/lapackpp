#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>
#include <iostream>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_ggev_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Job jobvl = params.jobvl.value();
    lapack::Job jobvr = params.jobvr.value();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();
    params.matrix.mark();
    params.matrixB.mark();

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

    lapack::generate_matrix( params.matrix,  n, n, nullptr, &A_tst[0], lda );
    lapack::generate_matrix( params.matrixB, n, n, nullptr, &B_tst[0], ldb );
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
