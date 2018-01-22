#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "check_geev.hh"
#include "matrix_opts.hh"
#include "matrix_generator.hh"

#include <vector>
#include <algorithm>

// -----------------------------------------------------------------------------
// comparison operator for sorting
template< typename T >
bool lessthan( T a, T b )
{
    return (real(a) < real(b)) || (real(a) == real(b) && imag(a) < imag(b));
}

// -----------------------------------------------------------------------------
// simple overloaded wrappers around LAPACKE
static lapack_int LAPACKE_geev(
    char jobvl, char jobvr, lapack_int n, float* A, lapack_int lda, std::complex<float>* W, float* VL, lapack_int ldvl, float* VR, lapack_int ldvr )
{
    std::vector< float > WR( n ), WI( n );
    lapack_int err = LAPACKE_sgeev( LAPACK_COL_MAJOR, jobvl, jobvr, n, A, lda, &WR[0], &WI[0], VL, ldvl, VR, ldvr );
    for (int64_t i = 0; i < n; ++i) {
        W[i] = std::complex<float>( WR[i], WI[i] );
    }
    return err;
}

static lapack_int LAPACKE_geev(
    char jobvl, char jobvr, lapack_int n, double* A, lapack_int lda, std::complex<double>* W, double* VL, lapack_int ldvl, double* VR, lapack_int ldvr )
{
    std::vector< double > WR( n ), WI( n );
    lapack_int err = LAPACKE_dgeev( LAPACK_COL_MAJOR, jobvl, jobvr, n, A, lda, &WR[0], &WI[0], VL, ldvl, VR, ldvr );
    for (int64_t i = 0; i < n; ++i) {
        W[i] = std::complex<double>( WR[i], WI[i] );
    }
    return err;
}

static lapack_int LAPACKE_geev(
    char jobvl, char jobvr, lapack_int n, std::complex<float>* A, lapack_int lda, std::complex<float>* W, std::complex<float>* VL, lapack_int ldvl, std::complex<float>* VR, lapack_int ldvr )
{
    return LAPACKE_cgeev( LAPACK_COL_MAJOR, jobvl, jobvr, n, A, lda, W, VL, ldvl, VR, ldvr );
}

static lapack_int LAPACKE_geev(
    char jobvl, char jobvr, lapack_int n, std::complex<double>* A, lapack_int lda, std::complex<double>* W, std::complex<double>* VL, lapack_int ldvl, std::complex<double>* VR, lapack_int ldvr )
{
    return LAPACKE_zgeev( LAPACK_COL_MAJOR, jobvl, jobvr, n, A, lda, W, VL, ldvl, VR, ldvr );
}

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_geev_work( Params& params, bool run )
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
    int64_t verbose = params.verbose.value();

    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol.value() * eps;
    //printf( "eps %.2e, tol %.2e, tol*eps %.2e\n", eps, params.tol.value(), tol );

    // mark non-standard output values
    params.error2.value();
    params.error3.value();
    params.error4.value();
    params.error5.value();
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();
    params.test_matrix_name.value();
    params.test_matrix_cond.value();
    params.test_matrix_condD.value();

    params.error .name( "A' Vl-Vl W'\nerror" );
    params.error2.name( "Vl(j) norm\nerror" );
    params.error3.name( "A Vr-Vr W\nerror" );
    params.error4.name( "Vr(j) norm\nerror" );
    params.error5.name( "W - Wref\nerror" );

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( max( 1, n ), align );
    int64_t ldvl = roundup( n, align );
    int64_t ldvr = roundup( n, align );
    size_t size_A = (size_t) lda * n;
    size_t size_W = (size_t) (n);
    size_t size_VL = (size_t) ldvl * n;
    size_t size_VR = (size_t) ldvr * n;

    std::vector< scalar_t > A_tst( size_A );
    std::vector< scalar_t > A_ref( size_A );
    std::vector< std::complex<real_t> > W_tst( size_W );
    std::vector< std::complex<real_t> > W_ref( size_W );
    std::vector< scalar_t > VL_tst( size_VL );
    std::vector< scalar_t > VL_ref( size_VL );
    std::vector< scalar_t > VR_tst( size_VR );
    std::vector< scalar_t > VR_ref( size_VR );

    // Generate test matrix
    Vector<real_t> sigma;
    matrix_opts opts;
    opts.matrix = params.test_matrix_name.value();
    opts.cond = params.test_matrix_cond.value();
    opts.condD = params.test_matrix_condD.value();

    Matrix<scalar_t> A( &A_tst[0], n, n, lda );
    lapack_generate_matrix( opts, sigma, A );

    std::copy( A(0, 0), A(n, 0), A_tst.begin() );
    A_ref = A_tst;

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld\n",
                (lld) n, (lld) lda );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( n, n, &A_tst[0], lda );
    }

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::geev( jobvl, jobvr, n, &A_tst[0], lda, &W_tst[0], &VL_tst[0], ldvl, &VR_tst[0], ldvr );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::geev returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::geev( jobvl, jobvr, n );
    //params.gflops.value() = gflop / time;

    if (verbose >= 2) {
        printf( "W = " ); print_vector( n, &W_tst[0], 1 );
        if (jobvl == lapack::Job::Vec) {
            printf( "VL = " ); print_matrix( n, n, &VL_tst[0], ldvl );
        }
        if (jobvr == lapack::Job::Vec) {
            printf( "VR = " ); print_matrix( n, n, &VR_tst[0], ldvr );
        }
    }

    bool okay = true;
    if (params.check.value() == 'y') {
        // ---------- check numerical error
        // formula from get22; differs from LAWN 41, usess ||V||_1 instead of n
        // 1. || A^H Vl - Vl W^H ||_1 / (||V||_1 ||A||_1)
        // 2. max_{j=1, ..., n} | || Vl(j) ||_2 - 1 |
        // 3. || A Vr - Vr W || / (||V||_1 ||A||_1)
        // 4. max_{j=1, ..., n} | || Vr(j) ||_2 - 1 |
        real_t results[4] = { real_t( libtest::no_data_flag ),
                              real_t( libtest::no_data_flag ),
                              real_t( libtest::no_data_flag ),
                              real_t( libtest::no_data_flag ) };
        if (jobvl == lapack::Job::Vec) {
            check_geev( Op::ConjTrans, n, &A_ref[0], lda, &W_tst[0],
                        &VL_tst[0], ldvl, verbose, &results[0] );
            okay = (okay && results[0] < tol && results[1] < tol);
            params.error .value() = results[0];
            params.error2.value() = results[1];
        }
        if (jobvr == lapack::Job::Vec) {
            check_geev( Op::NoTrans,   n, &A_ref[0], lda, &W_tst[0],
                        &VR_tst[0], ldvr, verbose, &results[2] );
            okay = (okay && results[2] < tol && results[3] < tol);
            params.error3.value() = results[2];
            params.error4.value() = results[3];
        }
    }

    if (params.ref.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_geev( job2char(jobvl), job2char(jobvr), n, &A_ref[0], lda, &W_ref[0], &VL_ref[0], ldvl, &VR_ref[0], ldvr );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_geev returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        if (verbose >= 2) {
            printf( "// note: may be sorted differently than results above\n" );
            printf( "Wref  = " ); print_vector( n, &W_ref[0], 1 );
            if (jobvl == lapack::Job::Vec) {
                printf( "VLref = " ); print_matrix( n, n, &VL_ref[0], ldvl );
            }
            if (jobvr == lapack::Job::Vec) {
                printf( "VRref = " ); print_matrix( n, n, &VR_ref[0], ldvr );
            }
        }

        // ---------- check error compared to reference
        // sort eigenvalues into lexical order for comparison
        real_t error = 0;
        std::sort( W_tst.begin(), W_tst.end(), lessthan< std::complex<real_t> > );
        std::sort( W_ref.begin(), W_ref.end(), lessthan< std::complex<real_t> > );
        error = rel_error( W_tst, W_ref );
        okay = (okay && error < tol);
        params.error5.value() = error;
    }

    // okay from error ... error5
    params.okay.value() = okay;
}

// -----------------------------------------------------------------------------
void test_geev( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_geev_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_geev_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_geev_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_geev_work< std::complex<double> >( params, run );
            break;
    }
}
