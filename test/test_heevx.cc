#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_heevx_work( Params& params, bool run )
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

    lapack::Range range;
    real_t  vl;  // = params.vl.value();
    real_t  vu;  // = params.vu.value();
    int64_t il;  // = params.il.value();
    int64_t iu;  // = params.iu.value();
    params.get_range( n, &range, &vl, &vu, &il, &iu );
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    real_t abstol = 0;   // use default
    int64_t nfound_tst;  // i.e., "m" in LAPACK
    int     nfound_ref;
    int64_t ldz = roundup( max( 1, n ), align );
    size_t size_A = (size_t) lda * n;
    size_t size_W = (size_t) (n);
    size_t size_Z = (size_t) ldz * max(1,n);
    size_t size_ifail = (size_t) (n);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< real_t > W_tst( size_W );
    std::vector< real_t > W_ref( size_W );
    std::vector< scalar_t > Z_tst( size_Z );
    std::vector< scalar_t > Z_ref( size_Z );
    std::vector< int64_t > ifail_tst( size_ifail );
    std::vector< lapack_int > ifail_ref( size_ifail );

    lapack::generate_matrix( params.matrix, n, n, nullptr, &A_tst[0], lda );
    A_ref = A_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::heevx( jobz, range, uplo, n, &A_tst[0], lda, vl, vu, il, iu, abstol, &nfound_tst, &W_tst[0], &Z_tst[0], ldz, &ifail_tst[0] );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::heevx returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::heevx( jobz, range, n );
    //params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_heevx( job2char(jobz), range2char(range), uplo2char(uplo), n, &A_ref[0], lda, vl, vu, il, iu, abstol, &nfound_ref, &W_ref[0], &Z_ref[0], ldz, &ifail_ref[0] );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_heevx returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += std::abs( nfound_tst - nfound_ref );
        error += abs_error( W_tst, W_ref );
        error += abs_error( Z_tst, Z_ref );
        // for ifail, just compare the first nfound values
        for ( size_t i = 0; i < (size_t)nfound_tst; i++ )
            error += std::abs( ifail_tst[i] - ifail_ref[i] );

        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_heevx( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_heevx_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_heevx_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_heevx_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_heevx_work< std::complex<double> >( params, run );
            break;
    }
}
