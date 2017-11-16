#ifndef TEST_HH
#define TEST_HH

#include "libtest.hh"
#include "lapack_util.hh"

// -----------------------------------------------------------------------------
class Params: public libtest::ParamsBase
{
public:
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double pi  = 3.141592653589793;
    const double e   = 2.718281828459045;

    Params();

    // Field members are explicitly public.
    // Order here determines output order.
    // ----- test framework parameters
    libtest::ParamChar   check;
    libtest::ParamChar   error_exit;
    libtest::ParamChar   ref;
    libtest::ParamDouble tol;
    libtest::ParamInt    repeat;
    libtest::ParamInt    verbose;
    libtest::ParamInt    cache;

    // ----- routine parameters
    libtest::ParamEnum< libtest::DataType > datatype;
    libtest::ParamEnum< lapack::Side >      side;
    libtest::ParamEnum< lapack::Uplo >      uplo;
    libtest::ParamEnum< lapack::Op >        trans;
    libtest::ParamEnum< lapack::Op >        transA;
    libtest::ParamEnum< lapack::Op >        transB;
    libtest::ParamEnum< lapack::Diag >      diag;

    libtest::ParamInt3   dim;
    libtest::ParamDouble alpha;
    libtest::ParamDouble beta;
    libtest::ParamInt    incx;
    libtest::ParamInt    incy;
    libtest::ParamInt    align;

    // ----- output parameters
    libtest::ParamScientific error;
    libtest::ParamScientific ortho;
    libtest::ParamDouble     time;
    libtest::ParamDouble     gflops;
    libtest::ParamInt        iters;

    libtest::ParamScientific ref_error;
    libtest::ParamScientific ref_ortho;
    libtest::ParamDouble     ref_time;
    libtest::ParamDouble     ref_gflops;
    libtest::ParamInt        ref_iters;

    libtest::ParamOkay       okay;
};

// -----------------------------------------------------------------------------
// LAPACK
void test_getrf ( Params& params, bool run );

// -----------------------------------------------------------------------------
template< typename T >
inline T roundup( T x, T y )
{
    return T( (x + y - 1) / y ) * y;
}

// -----------------------------------------------------------------------------
#define assert_throw( expr, exception_type ) \
    try { \
        expr; \
        fprintf( stderr, "Error: didn't throw expected exception at %s:%d\n", \
                 __FILE__, __LINE__ ); \
        throw std::exception(); \
    } \
    catch (exception_type& err) { \
        if (verbose >= 3) { \
            printf( "Caught expected exception: %s\n", err.what() ); \
        } \
    }

#endif  //  #ifndef TEST_HH
