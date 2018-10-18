#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_hbgvx_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Job jobz = params.jobz();
    lapack::Uplo uplo = params.uplo();
    int64_t n = params.dim.n();
    int64_t ka = params.kd();
    int64_t kb = params.kd();
    int64_t align = params.align();

    real_t  vl;  // = params.vl();
    real_t  vu;  // = params.vu();
    int64_t il;  // = params.il();
    int64_t iu;  // = params.iu();
    lapack::Range range;  // derived from vl,vu,il,iu
    params.get_range( n, &range, &vl, &vu, &il, &iu );

    // mark non-standard output values
    params.ref_time();
    // params.ref_gflops();
    // params.gflops();

    if (! run)
        return;

    // ---------- setup
    int64_t ldab = roundup( ka+1, align );
    int64_t ldbb = roundup( kb+1, align );
    int64_t ldq = roundup( max( 1, n ), align );
    real_t abstol = 0;  // default value
    int64_t m_tst;
    lapack_int m_ref;
    int64_t ldz = ( jobz==lapack::Job::Vec ? roundup( max( 1, n ), align ) : 1 );
    size_t size_AB = (size_t) ( ldab * n );
    size_t size_BB = (size_t) ( ldbb * n );
    size_t size_Q = (size_t) ( ldq * n );
    size_t size_W = (size_t) (n);
    size_t size_Z = (size_t) ( ldz * n );
    size_t size_ifail = (size_t) (n);

    std::vector< scalar_t > AB_tst( size_AB );
    std::vector< scalar_t > AB_ref( size_AB );
    std::vector< scalar_t > BB_tst( size_BB );
    std::vector< scalar_t > BB_ref( size_BB );
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
    lapack::larnv( idist, iseed, BB_tst.size(), &BB_tst[0] );

    // diagonally dominant -> positive definite
    if (uplo == lapack::Uplo::Upper) {
        for (int64_t j = 0; j < n; ++j) {
            BB_tst[ kb + j*ldbb ] += n;
        }
    } else { // lower
       for (int64_t j = 0; j < n; ++j) {
           BB_tst[ j*ldbb ] += n;
       }
    }

    AB_ref = AB_tst;
    BB_ref = BB_tst;

    // ---------- run test
    libtest::flush_cache( params.cache() );
    double time = get_wtime();
    int64_t info_tst = lapack::hbgvx( jobz, range, uplo, n, ka, kb, &AB_tst[0], ldab, &BB_tst[0], ldbb, &Q_tst[0], ldq, vl, vu, il, iu, abstol, &m_tst, &W_tst[0], &Z_tst[0], ldz, &ifail_tst[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::hbgvx returned error %lld\n", (lld) info_tst );
    }

    params.time() = time;
    // double gflop = lapack::Gflop< scalar_t >::hbgvx( jobz, range, n, ka, kb );
    // params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_hbgvx( job2char(jobz), range2char(range), uplo2char(uplo), n, ka, kb, &AB_ref[0], ldab, &BB_ref[0], ldbb, &Q_ref[0], ldq, vl, vu, il, iu, abstol, &m_ref, &W_ref[0], &Z_ref[0], ldz, &ifail_ref[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_hbgvx returned error %lld\n", (lld) info_ref );
        }

        params.ref_time() = time;
        // params.ref_gflops() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( AB_tst, AB_ref );
        error += abs_error( BB_tst, BB_ref );
        error += abs_error( Q_tst, Q_ref );
        error += std::abs( m_tst - m_ref );
        error += abs_error( W_tst, W_ref );
        error += abs_error( Z_tst, Z_ref );
        // Check first m elements of ifail
        if ( jobz==lapack::Job::Vec ) {
            for ( size_t i = 0; i < (size_t)(m_ref); i++ )
                error += std::abs( ifail_tst[i] - ifail_ref[i] );
        }
        params.error() = error;
        params.okay() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_hbgvx( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_hbgvx_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_hbgvx_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_hbgvx_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_hbgvx_work< std::complex<double> >( params, run );
            break;
    }
}
