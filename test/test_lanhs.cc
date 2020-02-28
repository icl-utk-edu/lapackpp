#include "test.hh"
#include "lapack.hh"
#include "lapack/flops.hh"
#include "print_matrix.hh"
#include "error.hh"
#include "lapacke_wrappers.hh"

#include <vector>

// -----------------------------------------------------------------------------
template< typename scalar_t >
void test_lanhs_work( Params& params, bool run )
{
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    lapack::Norm norm = params.norm.value();
    int64_t n = params.dim.n();
    int64_t verbose = params.verbose();
    int64_t align = params.align.value();

    // mark non-standard output values
    params.ref_time.value();
    //params.ref_gflops.value();
    //params.gflops.value();

    if (! run)
        return;

    // ---------- setup
    int64_t lda = roundup( blas::max( n, 1 ), align );
    size_t size_A = (size_t) lda * n;

    std::vector< scalar_t > A( size_A );

    lapack::generate_matrix( params.matrix, n, n, &A[0], lda );

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld\n",
                (lld) n, (lld) lda );
    }
    if (verbose >= 2) {
        printf( "A = " ); print_matrix( n, n, &A[0], lda );
    }

    // ---------- run test
    testsweeper::flush_cache( params.cache.value() );
    double time = testsweeper::get_wtime();
    real_t norm_tst = lapack::lanhs( norm, n, &A[0], lda );
    time = testsweeper::get_wtime() - time;

    params.time.value() = time;
    //double gflop = lapack::Gflop< scalar_t >::lanhs( norm, n );
    //params.gflops.value() = gflop / time;

    if (verbose >= 1) {
        printf( "norm_tst = %.8e\n", norm_tst );
    }

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // ---------- run reference
        testsweeper::flush_cache( params.cache.value() );
        time = testsweeper::get_wtime();
        real_t norm_ref = LAPACKE_lanhs( norm2char(norm), n, &A[0], lda );
        time = testsweeper::get_wtime() - time;

        params.ref_time.value() = time;
        //params.ref_gflops.value() = gflop / time;

        if (verbose >= 1) {
            printf( "norm_ref = %.8e\n", norm_ref );
        }

        // ---------- check error compared to reference
        real_t tol = 3 * std::numeric_limits< real_t >::epsilon();
        real_t normalize = 1;
        if (norm == lapack::Norm::Max && ! blas::is_complex< scalar_t >::value) {
            // max-norm depends on only one element, so in real there should be
            // zero error, but in complex there's error in abs().
            tol = 0;
        }
        else if (norm == lapack::Norm::One)
            normalize = sqrt( real_t(n) );
        else if (norm == lapack::Norm::Inf)
            normalize = sqrt( real_t(n) );
        else if (norm == lapack::Norm::Fro)
            normalize = sqrt( real_t(n)*n );
        real_t error = std::abs( norm_tst - norm_ref ) / normalize;
        if (norm_ref != 0)
            error /= norm_ref;
        params.error() = error;
        params.okay() = (error <= tol);
    }
}

// -----------------------------------------------------------------------------
void test_lanhs( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case testsweeper::DataType::Integer:
            throw std::exception();
            break;

        case testsweeper::DataType::Single:
            test_lanhs_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_lanhs_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_lanhs_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_lanhs_work< std::complex<double> >( params, run );
            break;
    }
}
