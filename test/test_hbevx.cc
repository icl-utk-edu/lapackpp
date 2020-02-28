#include "test.hh"
#include "lapack.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_hbevx_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Job jobz = params.jobz();
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t kd = params.kd();
    int64_t align = params.align();

    lapack::Range range;
    real_t  vl;  // = params.vl();
    real_t  vu;  // = params.vu();
    int64_t il;  // = params.il();
    int64_t iu;  // = params.iu();
    params.get_range( n, &range, &vl, &vu, &il, &iu );

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( kd + 1, align );
    int64_t ldq = roundup( blas::max( 1, n ), align );
    real_t abstol = 0; // use default
    int64_t m_tst;
    lapack_int m_ref;
    int64_t ldz = ( jobz == lapack::Job::NoVec ? 1: roundup( blas::max( 1, n ), align ) );
    size_t size_AB = (size_t) ldab * n;
    size_t size_Q = (size_t) ldq * n;
    size_t size_W = (size_t) (n);
    size_t size_Z = (size_t) ldz * blas::max(1,n);
    size_t size_ifail = (size_t) (n);

    std::vector< scalar_t > AB_tst( size_AB );
    std::vector< scalar_t > AB_ref( size_AB );
    std::vector< scalar_t > Q_tst( size_Q );
    std::vector< scalar_t > Q_ref( size_Q );
    std::vector< real_t > W_tst( size_W );
    std::vector< real_t > W_ref( size_W );
    std::vector< scalar_t > Z_tst( size_Z );
    std::vector< scalar_t > Z_ref( size_Z );
    std::vector< int64_t > ifail_tst( size_ifail );
    std::vector< lapack_int > ifail_ref( size_ifail );

    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, AB_tst.size(), &AB_tst[0] );
    AB_ref = AB_tst;

    // ---------- run test
    testsweeper::flush_cache( params.cache() );
    double time = testsweeper::get_wtime();
    int64_t info_tst = lapack::hbevx( jobz, range, uplo, n, kd, &AB_tst[0], ldab, &Q_tst[0], ldq, vl, vu, il, iu, abstol, &m_tst, &W_tst[0], &Z_tst[0], ldz, &ifail_tst[0] );
    time = testsweeper::get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::hbevx returned error %lld\n", (lld) info_tst );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::hbevx( jobz, range, n, kd );
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache() );
        time = testsweeper::get_wtime();
        int64_t info_ref = LAPACKE_hbevx( job2char(jobz), range2char(range), uplo2char(uplo), n, kd, &AB_ref[0], ldab, &Q_ref[0], ldq, vl, vu, il, iu, abstol, &m_ref, &W_ref[0], &Z_ref[0], ldz, &ifail_ref[0] );
        time = testsweeper::get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_hbevx returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( AB_tst, AB_ref );
        error += abs_error( Q_tst, Q_ref );
        error += std::abs( m_tst - m_ref );
        error += abs_error( W_tst, W_ref );
        error += abs_error( Z_tst, Z_ref );
        // for ifail, just compare the first m_tst values
        for ( size_t i = 0; i < (size_t)m_tst; i++ )
            error += std::abs( ifail_tst[i] - ifail_ref[i] );
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_hbevx( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_hbevx_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_hbevx_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_hbevx_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_hbevx_work< std::complex<double> >( params, run );
            break;
    }
}
