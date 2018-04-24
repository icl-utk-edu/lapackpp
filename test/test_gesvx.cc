#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gesvx_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Factored fact = params.factored.value();
    lapack::Op trans = params.trans.value();
    lapack::Equed equed = params.equed.value();
    int64_t n = params.dim.n();
    int64_t nrhs = params.nrhs.value();
    int64_t align = params.align.value();
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time.value();
    // params.ref_gflops.value();
    // params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    int64_t ldaf = roundup( max( 1, n ), align );
    int64_t ldb = roundup( max( 1, n ), align );
    int64_t ldx = roundup( max( 1, n ), align );
    real_t rcond_tst = 0;
    real_t rcond_ref = 0; 
    real_t rpivot_tst = 0;
    real_t rpivot_ref = 0;
    lapack::Equed equed_tst = ( fact==lapack::Factored::Factored ? equed : lapack::Equed::None );
    lapack::Equed equed_ref = equed_tst;
    size_t size_A = (size_t) lda * n;
    size_t size_AF = (size_t) ldaf * n;
    size_t size_ipiv = (size_t) (n);
    size_t size_R = (size_t) (n);
    size_t size_C = (size_t) (n);
    size_t size_B = (size_t) ldb * nrhs;
    size_t size_X = (size_t) ldx * nrhs;
    size_t size_ferr = (size_t) (nrhs);
    size_t size_berr = (size_t) (nrhs);

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< scalar_t > AF_tst( size_AF );
    std::vector< scalar_t > AF_ref( size_AF );
    std::vector< int64_t > ipiv_tst( size_ipiv );
    std::vector< lapack_int > ipiv_ref( size_ipiv );
    std::vector< real_t > R_tst( size_R );
    std::vector< real_t > R_ref( size_R );
    std::vector< real_t > C_tst( size_C );
    std::vector< real_t > C_ref( size_C );
    std::vector< scalar_t > B_tst( size_B );
    std::vector< scalar_t > B_ref( size_B );
    std::vector< scalar_t > X_tst( size_X );
    std::vector< scalar_t > X_ref( size_X );
    std::vector< real_t > ferr_tst( size_ferr );
    std::vector< real_t > ferr_ref( size_ferr );
    std::vector< real_t > berr_tst( size_berr );
    std::vector< real_t > berr_ref( size_berr );

    lapack::generate_matrix( params.matrix, n, n, nullptr, &A_tst[0], lda );
    int64_t idist = 1;
    int64_t iseed[4] = { 0, 1, 2, 3 };
    lapack::larnv( idist, iseed, R_tst.size(), &R_tst[0] );
    lapack::larnv( idist, iseed, C_tst.size(), &C_tst[0] );
    lapack::larnv( idist, iseed, B_tst.size(), &B_tst[0] );

    // Factor A using copy AF to initialize ipiv_tst and ipiv_ref and AF
    AF_tst = A_tst;
    int64_t info_trf = lapack::getrf( n, n, &AF_tst[0], lda, &ipiv_tst[0] );
    if (info_trf != 0) {
        fprintf( stderr, "lapack::getrf returned error %lld\n", (lld) info_trf );
    }
    std::copy( ipiv_tst.begin(), ipiv_tst.end(), ipiv_ref.begin() );

    A_ref = A_tst;
    AF_ref = AF_tst;
    R_ref = R_tst;
    C_ref = C_tst;
    B_ref = B_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::gesvx( fact, trans, n, nrhs, &A_tst[0], lda, &AF_tst[0], ldaf, &ipiv_tst[0], &equed_tst, &R_tst[0], &C_tst[0], &B_tst[0], ldb, &X_tst[0], ldx, &rcond_tst, &ferr_tst[0], &berr_tst[0], &rpivot_tst );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gesvx returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    // double gflop = lapack::Gflop< scalar_t >::gesvx( fact, trans, n, nrhs );
    // params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        char equed_ref_char = lapack::equed2char( equed_ref );
        time = get_wtime();
        int64_t info_ref = LAPACKE_gesvx( factored2char(fact), op2char(trans), n, nrhs, &A_ref[0], lda, &AF_ref[0], ldaf, &ipiv_ref[0], &equed_ref_char, &R_ref[0], &C_ref[0], &B_ref[0], ldb, &X_ref[0], ldx, &rcond_ref, &ferr_ref[0], &berr_ref[0], &rpivot_ref );
        time = get_wtime() - time;
        equed_ref = lapack::char2equed( equed_ref_char );
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gesvx returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        // params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        real_t error = 0;
        if (info_tst != info_ref) {
            error = 1;
        }
        error += abs_error( A_tst, A_ref );
        error += abs_error( AF_tst, AF_ref );
        error += abs_error( ipiv_tst, ipiv_ref );
        error += ( equed_tst != equed_ref ? 1 : 0 );
        error += abs_error( R_tst, R_ref );
        error += abs_error( C_tst, C_ref );
        error += abs_error( B_tst, B_ref );
        error += abs_error( X_tst, X_ref );
        error += std::abs( rcond_tst - rcond_ref );
        error += abs_error( ferr_tst, ferr_ref );
        error += abs_error( berr_tst, berr_ref );
        error += std::abs( rpivot_tst - rpivot_ref );
        params.error.value() = error;
        params.okay.value() = (error == 0);  // expect lapackpp == lapacke
    }
}

// -----------------------------------------------------------------------------
void test_gesvx( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gesvx_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gesvx_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gesvx_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gesvx_work< std::complex<double> >( params, run );
            break;
    }
}
