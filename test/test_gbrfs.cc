#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gbrfs_work( Params& params, bool run )
{
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Op trans = params.trans();
    int64_t n = params.dim.n();
    int64_t kl = params.kl();
    int64_t ku = params.ku();
    int64_t nrhs = params.nrhs();
    int64_t align = params.align();

    // mark non-standard output values
    params.ref_time();
    //params.ref_gflops();
    //params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( kl+ku+1, align );
    int64_t ldafb = roundup( 2*kl*ku+1, align );
    int64_t ldb = roundup( max( 1, n ), align );
    int64_t ldx = roundup( max( 1, n ), align );
    size_t size_AB = (size_t) ldab * n;
    size_t size_AFB = (size_t) ldafb * n;
    size_t size_ipiv = (size_t) (n);
    size_t size_B = (size_t) ldb * nrhs;
    size_t size_X = (size_t) ldx * nrhs;
    size_t size_ferr = (size_t) (nrhs);
    size_t size_berr = (size_t) (nrhs);

    std::vector< scalar_t > AB( size_AB );
    std::vector< scalar_t > AFB( size_AFB );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );
    std::vector< scalar_t > B( size_B );
    std::vector< scalar_t > X_tst( size_X );
    std::vector< scalar_t > X_ref( size_X );
    std::vector< real_t > ferr_tst( size_ferr );
    std::vector< real_t > ferr_ref( size_ferr );
    std::vector< real_t > berr_tst( size_berr );
    std::vector< real_t > berr_ref( size_berr );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB.size(), &AB[0] );
    lapack::larnv( idist, iseed, AFB.size(), &AFB[0] );
    // todo: initialize ipiv_tst and ipiv_ref
    lapack::larnv( idist, iseed, B.size(), &B[0] );
    lapack::larnv( idist, iseed, X_tst.size(), &X_tst[0] );
    X_ref = X_tst;

    AFB = AB;
    int64_t info = lapack::gbtrf( n, n, kl, ku, &AFB[0], ldafb, &ipiv_tst[0] );
    if (info != 0) {
        fprintf( stderr, "lapack::gbtrf returned error %lld\n", (lld) info );
    }

    info = lapack::gbtrs( trans, n, kl, ku, nrhs, &AFB[0], ldafb, &ipiv_tst[0], &B[0], ldb );
    if (info != 0) {
        fprintf( stderr, "lapack::gbtrs returned error %lld\n", (lld) info );
    }

    std::copy (ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = libtest::get_wtime();
    int64_t info_tst = lapack::gbrfs( trans, n, kl, ku, nrhs, &AB[0], ldab, &AFB[0], ldafb, &ipiv_tst[0], &B[0], ldb, &X_tst[0], ldx, &ferr_tst[0], &berr_tst[0] );
    time = libtest::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gbrfs returned error %lld\n", (lld) info_tst );
    }

    params.time() = time;
    //double gflop = lapack::Gflop< scalar_t >::gbrfs( trans, n, kl, ku, nrhs );
    //params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = libtest::get_wtime();
        int64_t info_ref = LAPACKE_gbrfs( op2char(trans), n, kl, ku, nrhs, &AB[0], ldab, &AFB[0], ldafb, &ipiv_ref[0], &B[0], ldb, &X_ref[0], ldx, &ferr_ref[0], &berr_ref[0] );
        time = libtest::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gbrfs returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        //params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( X_tst, X_ref );
        error += abs_error( ferr_tst, ferr_ref );
        error += abs_error( berr_tst, berr_ref );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gbrfs( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gbrfs_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gbrfs_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gbrfs_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gbrfs_work< std::complex<double> >( params, run );
            break;
    }
}
