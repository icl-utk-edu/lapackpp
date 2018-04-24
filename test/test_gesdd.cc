#include "test.hh"
#include "lapack.hh"
#include "lapack_flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"
#include "check_svd.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_gesdd_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using namespace lapack;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Job jobu = params.jobu.value();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align.value();
    params.matrix.mark();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();
    params.ortho_U.value();
    params.ortho_V.value();
    params.error_sigma.value();

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

    lapack::generate_matrix( params.matrix, m, n, nullptr, &A_tst[0], lda );
    A_ref = A_tst;

    // ---------- run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    int64_t info_tst = lapack::gesdd( jobu, m, n, &A_tst[0], lda, &S_tst[0], &U_tst[0], ldu, &VT_tst[0], ldvt );
    time = get_wtime() - time;
    if (info_tst != 0) {
        fprintf( stderr, "lapack::gesdd returned error %lld\n", (lld) info_tst );
    }

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::gesdd( jobu, m, n );
    //params.gflops.value() = gflop / time;

    // ---------- check numerical error
    // errors[0] = || A - U diag(S) VT || / (||A|| max(m,n)),
    //                                    if jobu  != NoVec
    // errors[1] = || I - U^H U || / m,   if jobu  != NoVec
    // errors[2] = || I - VT VT^H || / n, if jobu  != NoVec
    // errors[3] = 0 if S has non-negative values in non-increasing order, else 1
    real_t errors[4] = { (real_t) libtest::no_data_flag,
                         (real_t) libtest::no_data_flag,
                         (real_t) libtest::no_data_flag,
                         (real_t) libtest::no_data_flag };
    if (params.check.value() == 'y') {
        // U2 or VT2 points to A if overwriting
        scalar_t* U2    = &U_tst[0];
        int64_t   ldu2  = ldu;
        scalar_t* VT2   = &VT_tst[0];
        int64_t   ldvt2 = ldvt;
        if (jobu == lapack::Job::OverwriteVec) {
            if (m >= n) {
                U2   = &A_tst[0];
                ldu2 = lda;
            }
            else {
                VT2   = &A_tst[0];
                ldvt2 = lda;
            }
        }
        check_svd( jobu, jobu, m, n, &A_ref[0], lda,
                   &S_tst[0], U2, ldu2, VT2, ldvt2, errors );
    }

    if (params.ref.value() == 'y') {
        // ---------- run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        int64_t info_ref = LAPACKE_gesdd( job2char(jobu), m, n, &A_ref[0], lda, &S_ref[0], &U_ref[0], ldu, &VT_ref[0], ldvt );
        time = get_wtime() - time;
        if (info_ref != 0) {
            fprintf( stderr, "LAPACKE_gesdd returned error %lld\n", (lld) info_ref );
        }

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        // ---------- check error compared to reference
        if (info_tst != info_ref) {
            errors[0] = 1;
        }
        errors[3] += rel_error( S_tst, S_ref );
    }
    real_t eps = std::numeric_limits< real_t >::epsilon();
    real_t tol = params.tol.value() * eps;
    params.error.value()       = errors[0];
    params.ortho_U.value()     = errors[1];
    params.ortho_V.value()     = errors[2];
    params.error_sigma.value() = errors[3];
    params.okay.value() = (
        (jobu == Job::NoVec || errors[0] < tol) &&
        (jobu == Job::NoVec || errors[1] < tol) &&
        (jobu == Job::NoVec || errors[2] < tol) &&
        errors[3] < tol);
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
