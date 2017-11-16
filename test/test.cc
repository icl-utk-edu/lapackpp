#include <complex>

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <omp.h>

#include "test.hh"

// -----------------------------------------------------------------------------
using libtest::ParamType;
using libtest::DataType;
using libtest::char2datatype;
using libtest::datatype2char;
using libtest::datatype2str;
using libtest::ansi_bold;
using libtest::ansi_red;
using libtest::ansi_normal;

// -----------------------------------------------------------------------------
enum Section {
    newline = 0,  // zero flag forces newline
    gesv,
    posv,
    sysv,
    sysv2,
    gels,
    qr,
    syev,
    sygv,
    geev,
    svd,
    aux,
    aux_norm,
    aux_householder,
    aux_gen,
};

const char* section_names[] = {
   "",  // none
   "LU",
   "Cholesky",
   "symmetric indefinite",
   "",
   "least squares",
   "QR, LQ, QL, RQ",
   "symmetric eigenvalues",
   "generalized symmetric eigenvalues",
   "non-symmetric eigenvalues",
   "singular value decomposition (SVD)",
   "auxiliary",
   "auxiliary - norms",
   "auxiliary - Householder",
   "auxiliary - matrix generation",
};

// { "", nullptr, Section::newline } entries force newline in help
std::vector< libtest::routines_t > routines = {
    // -----
    // LU
    { "gesv",               nullptr,        Section::gesv },
    { "gbsv",               nullptr,        Section::gesv },
    { "gtsv",               nullptr,        Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "getrf",              test_getrf,     Section::gesv },
    { "gbtrf",              nullptr,        Section::gesv },
    { "gttrf",              nullptr,        Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "getrs",              nullptr,        Section::gesv },
    { "gbtrs",              nullptr,        Section::gesv },
    { "gttrs",              nullptr,        Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "getri",              nullptr,        Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "gecon",              nullptr,        Section::gesv },
    { "gbcon",              nullptr,        Section::gesv },
    { "gtcon",              nullptr,        Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "gerfs",              nullptr,        Section::gesv },
    { "gbrfs",              nullptr,        Section::gesv },
    { "gtrfs",              nullptr,        Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "geequ",              nullptr,        Section::gesv },
    { "gbequ",              nullptr,        Section::gesv },
    { "gtequ",              nullptr,        Section::gesv },
    { "",                   nullptr,        Section::newline },

    // -----
    // Cholesky
    { "posv",               nullptr,        Section::posv },
    { "ppsv",               nullptr,        Section::posv },
    { "pbsv",               nullptr,        Section::posv },
    { "ptsv",               nullptr,        Section::posv },
    { "",                   nullptr,        Section::newline },

    { "potrf",              nullptr,        Section::posv },
    { "pptrf",              nullptr,        Section::posv },
    { "pbtrf",              nullptr,        Section::posv },
    { "pttrf",              nullptr,        Section::posv },
    { "",                   nullptr,        Section::newline },

    { "potrs",              nullptr,        Section::posv },
    { "pptrs",              nullptr,        Section::posv },
    { "pbtrs",              nullptr,        Section::posv },
    { "pttrs",              nullptr,        Section::posv },
    { "",                   nullptr,        Section::newline },

    { "potri",              nullptr,        Section::posv },
    { "pptri",              nullptr,        Section::posv },
    { "",                   nullptr,        Section::newline },

    { "pocon",              nullptr,        Section::posv },
    { "ppcon",              nullptr,        Section::posv },
    { "pbcon",              nullptr,        Section::posv },
    { "ptcon",              nullptr,        Section::posv },
    { "",                   nullptr,        Section::newline },

    { "porfs",              nullptr,        Section::posv },
    { "pprfs",              nullptr,        Section::posv },
    { "pbrfs",              nullptr,        Section::posv },
    { "ptrfs",              nullptr,        Section::posv },
    { "",                   nullptr,        Section::newline },

    { "poequ",              nullptr,        Section::posv },
    { "ppequ",              nullptr,        Section::posv },
    { "pbequ",              nullptr,        Section::posv },
    { "ptequ",              nullptr,        Section::posv },
    { "",                   nullptr,        Section::newline },

    // -----
    // symmetric indefinite
    { "sysv",               nullptr,        Section::sysv },
    { "spsv",               nullptr,        Section::sysv },
    { "",                   nullptr,        Section::newline },

    { "sytrf",              nullptr,        Section::sysv },
    { "sptrf",              nullptr,        Section::sysv },
    { "",                   nullptr,        Section::newline },

    { "sytrs",              nullptr,        Section::sysv },
    { "sptrs",              nullptr,        Section::sysv },
    { "",                   nullptr,        Section::newline },

    { "sytri",              nullptr,        Section::sysv },
    { "sptri",              nullptr,        Section::sysv },
    { "",                   nullptr,        Section::newline },

    { "sycon",              nullptr,        Section::sysv },
    { "spcon",              nullptr,        Section::sysv },
    { "",                   nullptr,        Section::newline },

    { "syrfs",              nullptr,        Section::sysv },
    { "sprfs",              nullptr,        Section::sysv },
    { "",                   nullptr,        Section::newline },

    // -----
    { "sysv_rook",          nullptr,        Section::sysv2 },
    { "sysv_aasen",         nullptr,        Section::sysv2 },
    { "sysv_aasen_2stage",  nullptr,        Section::sysv2 },
    { "",                   nullptr,        Section::newline },

    { "sytrf_rook",         nullptr,        Section::sysv2 },
    { "sytrf_aasen",        nullptr,        Section::sysv2 },
    { "sytrf_aasen_2stage", nullptr,        Section::sysv2 },
    { "",                   nullptr,        Section::newline },

    { "sytrs_rook",         nullptr,        Section::sysv2 },
    { "sytrs_aasen",        nullptr,        Section::sysv2 },
    { "sytrs_aasen_2stage", nullptr,        Section::sysv2 },
    { "",                   nullptr,        Section::newline },

    { "sytri_rook",         nullptr,        Section::sysv2 },
    { "sytri_aasen",        nullptr,        Section::sysv2 },
    { "sytri_aasen_2stage", nullptr,        Section::sysv2 },
    { "",                   nullptr,        Section::newline },

    // -----
    // least squares
    { "gels",               nullptr,        Section::gels },
    { "gelsy",              nullptr,        Section::gels },
    { "gelsd",              nullptr,        Section::gels },
    { "gelss",              nullptr,        Section::gels },
    { "getsls",             nullptr,        Section::gels },
    { "",                   nullptr,        Section::newline },

    { "gglse",              nullptr,        Section::gels },
    { "ggglm",              nullptr,        Section::gels },
    { "",                   nullptr,        Section::newline },

    // -----
    // QR, LQ, RQ, QL
    { "geqrf",              nullptr,        Section::qr },
    { "gelqf",              nullptr,        Section::qr },
    { "geqlf",              nullptr,        Section::qr },
    { "gerqf",              nullptr,        Section::qr },
    { "",                   nullptr,        Section::newline },

    { "ggqrf",              nullptr,        Section::qr },
    { "gglqf",              nullptr,        Section::qr },
    { "ggqlf",              nullptr,        Section::qr },
    { "ggrqf",              nullptr,        Section::qr },
    { "",                   nullptr,        Section::newline },

    { "ungqr",              nullptr,        Section::qr },
    { "unglq",              nullptr,        Section::qr },
    { "ungql",              nullptr,        Section::qr },
    { "ungrq",              nullptr,        Section::qr },
    { "",                   nullptr,        Section::newline },

    { "unmqr",              nullptr,        Section::qr },
    { "unmlq",              nullptr,        Section::qr },
    { "unmql",              nullptr,        Section::qr },
    { "unmrq",              nullptr,        Section::qr },
    { "",                   nullptr,        Section::newline },

    // -----
    // symmetric/Hermitian eigenvalues
    { "syev",               nullptr,        Section::syev },
    { "spev",               nullptr,        Section::syev },
    { "sbev",               nullptr,        Section::syev },
    { "",                   nullptr,        Section::newline },

    { "syevx",              nullptr,        Section::syev },
    { "spevx",              nullptr,        Section::syev },
    { "sbevx",              nullptr,        Section::syev },
    { "",                   nullptr,        Section::newline },

    { "syevd",              nullptr,        Section::syev },
    { "spevd",              nullptr,        Section::syev },
    { "sbevd",              nullptr,        Section::syev },
    { "",                   nullptr,        Section::newline },

    { "syevr",              nullptr,        Section::syev },
    { "spevr",              nullptr,        Section::syev },
    { "sbevr",              nullptr,        Section::syev },
    { "",                   nullptr,        Section::newline },

    { "sytrd",              nullptr,        Section::syev },
    { "sptrd",              nullptr,        Section::syev },
    { "sbtrd",              nullptr,        Section::syev },
    { "",                   nullptr,        Section::newline },

    { "orgtr",              nullptr,        Section::syev },
    { "opgtr",              nullptr,        Section::syev },
    { "obgtr",              nullptr,        Section::syev },
    { "",                   nullptr,        Section::newline },

    { "symtr",              nullptr,        Section::syev },
    { "spmtr",              nullptr,        Section::syev },
    { "sbmtr",              nullptr,        Section::syev },
    { "",                   nullptr,        Section::newline },

    // -----
    // generalized symmetric eigenvalues
    { "sygv",               nullptr,        Section::sygv },
    { "spgv",               nullptr,        Section::sygv },
    { "sbgv",               nullptr,        Section::sygv },
    { "",                   nullptr,        Section::newline },

    { "sygvx",              nullptr,        Section::sygv },
    { "spgvx",              nullptr,        Section::sygv },
    { "sbgvx",              nullptr,        Section::sygv },
    { "",                   nullptr,        Section::newline },

    { "sygvd",              nullptr,        Section::sygv },
    { "spgvd",              nullptr,        Section::sygv },
    { "sbgvd",              nullptr,        Section::sygv },
    { "",                   nullptr,        Section::newline },

    { "sygvr",              nullptr,        Section::sygv },
    { "spgvr",              nullptr,        Section::sygv },
    { "sbgvr",              nullptr,        Section::sygv },
    { "",                   nullptr,        Section::newline },

    { "sygst",              nullptr,        Section::sygv },
    { "spgst",              nullptr,        Section::sygv },
    { "sbgst",              nullptr,        Section::sygv },
    { "",                   nullptr,        Section::newline },

    // -----
    // non-symmetric eigenvalues
    { "geev",               nullptr,        Section::geev },
    { "ggev",               nullptr,        Section::geev },
    { "",                   nullptr,        Section::newline },

    { "geevx",              nullptr,        Section::geev },
    { "ggevx",              nullptr,        Section::geev },
    { "",                   nullptr,        Section::newline },

    { "gees",               nullptr,        Section::geev },
    { "gges",               nullptr,        Section::geev },
    { "",                   nullptr,        Section::newline },

    { "geesx",              nullptr,        Section::geev },
    { "ggesx",              nullptr,        Section::geev },
    { "",                   nullptr,        Section::newline },

    { "gehrd",              nullptr,        Section::geev },
    { "orghr",              nullptr,        Section::geev },
    { "ormhr",              nullptr,        Section::geev },
    { "hsein",              nullptr,        Section::geev },
    { "trevc",              nullptr,        Section::geev },
    { "",                   nullptr,        Section::newline },

    // -----
    // driver: singular value decomposition
    { "gesvd",              nullptr,        Section::svd },
    { "gesvd_2stage",       nullptr,        Section::svd },
    { "",                   nullptr,        Section::newline },

    { "gesdd",              nullptr,        Section::svd },
    { "gesdd_2stage",       nullptr,        Section::svd },
    { "",                   nullptr,        Section::newline },

    { "gesvdx",             nullptr,        Section::svd },
    { "gesvdx_2stage",      nullptr,        Section::svd },
    { "",                   nullptr,        Section::newline },

    { "gejsv",              nullptr,        Section::svd },
    { "gesvj",              nullptr,        Section::svd },

    // -----
    // auxiliary
    { "lacpy",              nullptr,        Section::aux },
    { "laswp",              nullptr,        Section::aux },
    { "laset",              nullptr,        Section::aux },

    // auxiliary: Householder
    { "larfg",              nullptr,        Section::aux_householder },
    { "larf",               nullptr,        Section::aux_householder },
    { "larfx",              nullptr,        Section::aux_householder },
    { "larfb",              nullptr,        Section::aux_householder },
    { "larft",              nullptr,        Section::aux_householder },

    // auxiliary: norms
    { "lange",              nullptr,        Section::aux_norm },
    { "lansy",              nullptr,        Section::aux_norm },
    { "lanhe",              nullptr,        Section::aux_norm },
    { "lantr",              nullptr,        Section::aux_norm },

    // auxiliary: matrix generation
    { "lagge",              nullptr,        Section::aux_gen },
    { "lagsy",              nullptr,        Section::aux_gen },
    { "laghe",              nullptr,        Section::aux_gen },
    { "lagtr",              nullptr,        Section::aux_gen },
};

// -----------------------------------------------------------------------------
// Params class
// List of parameters

Params::Params():
    ParamsBase(),

    // w = width
    // p = precision
    // def = default
    // ----- test framework parameters
    //         name,       w,    type,             def, valid, help
    check     ( "check",   0,    ParamType::Value, 'y', "ny",  "check the results" ),
    error_exit( "error-exit", 0, ParamType::Value, 'n', "ny",  "check error exits" ),
    ref       ( "ref",     0,    ParamType::Value, 'n', "ny",  "run reference; sometimes check implies ref" ),

    //          name,      w, p, type,             def, min,  max, help
    tol       ( "tol",     0, 0, ParamType::Value,  50,   1, 1000, "tolerance (e.g., error < tol*epsilon to pass)" ),
    repeat    ( "repeat",  0,    ParamType::Value,   1,   1, 1000, "number of times to repeat each test" ),
    verbose   ( "verbose", 0,    ParamType::Value,   0,   0,   10, "verbose level" ),
    cache     ( "cache",   0,    ParamType::Value,  20,   1, 1024, "total cache size, in MiB" ),

    // ----- routine parameters
    //          name,      w,    type,            def,                    char2enum,         enum2char,         enum2str,         help
    datatype  ( "type",    4,    ParamType::List, DataType::Double,       char2datatype,     datatype2char,     datatype2str,     "s=single (float), d=double, c=complex-single, z=complex-double" ),
    side      ( "side",    6,    ParamType::List, blas::Side::Left,       blas::char2side,   blas::side2char,   blas::side2str,   "side: l=left, r=right" ),
    uplo      ( "uplo",    6,    ParamType::List, blas::Uplo::Lower,      blas::char2uplo,   blas::uplo2char,   blas::uplo2str,   "triangle: l=lower, u=upper" ),
    trans     ( "trans",   7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose: n=no-trans, t=trans, c=conj-trans" ),
    transA    ( "transA",  7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose of A: n=no-trans, t=trans, c=conj-trans" ),
    transB    ( "transB",  7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose of B: n=no-trans, t=trans, c=conj-trans" ),
    diag      ( "diag",    7,    ParamType::List, blas::Diag::NonUnit,    blas::char2diag,   blas::diag2char,   blas::diag2str,   "diagonal: n=non-unit, u=unit" ),

    //          name,      w, p, type,            def,   min,     max, help
    dim       ( "dim",     6,    ParamType::List,          0, 1000000, "m x n x k dimensions" ),
    alpha     ( "alpha",   9, 4, ParamType::List,  pi,  -inf,     inf, "scalar alpha" ),
    beta      ( "beta",    9, 4, ParamType::List,   e,  -inf,     inf, "scalar beta" ),
    incx      ( "incx",    6,    ParamType::List,   1, -1000,    1000, "stride of x vector" ),
    incy      ( "incy",    6,    ParamType::List,   1, -1000,    1000, "stride of y vector" ),
    align     ( "align",   6,    ParamType::List,   1,     1,    1024, "column alignment (sets lda, ldb, etc. to multiple of align)" ),


    // ----- output parameters
    // min, max are ignored
    //          name,                  w, p, type,              def, min, max, help
    error     ( "LAPACK++\nerror",       11, 4, ParamType::Output, nan,   0,   0, "numerical error" ),
    ortho     ( "LAPACK++\north. error", 11, 4, ParamType::Output, nan,   0,   0, "orthogonality error" ),
    time      ( "LAPACK++\ntime (s)",    11, 4, ParamType::Output, nan,   0,   0, "time to solution" ),
    gflops    ( "LAPACK++\nGflop/s",     11, 4, ParamType::Output, nan,   0,   0, "Gflop/s rate" ),
    iters     ( "LAPACK++\niters",        6,    ParamType::Output,   0,   0,   0, "iterations to solution" ),

    ref_error ( "Ref.\nerror",        11, 4, ParamType::Output, nan,   0,   0, "reference numerical error" ),
    ref_ortho ( "Ref.\north. error",  11, 4, ParamType::Output, nan,   0,   0, "reference orthogonality error" ),
    ref_time  ( "Ref.\ntime (s)",     11, 4, ParamType::Output, nan,   0,   0, "reference time to solution" ),
    ref_gflops( "Ref.\nGflop/s",      11, 4, ParamType::Output, nan,   0,   0, "reference Gflop/s rate" ),
    ref_iters ( "Ref.\niters",         6,    ParamType::Output,   0,   0,   0, "reference iterations to solution" ),

    // default -1 means "no check"
    okay      ( "status",              6,    ParamType::Output,  -1,   0,   0, "success indicator" )
{
    // mark standard set of output fields as used
    okay  .value();
    error .value();
    time  .value();
    gflops.value();

    // mark framework parameters as used, so they will be accepted on the command line
    check  .value();
    error_exit.value();
    tol    .value();
    repeat .value();
    verbose.value();
    cache  .value();

    // routine's parameters are marked by the test routine; see main
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    // Usage: test routine [params]
    // find routine to test
    if (argc < 2 ||
        strcmp( argv[1], "-h" ) == 0 ||
        strcmp( argv[1], "--help" ) == 0)
    {
        usage( argc, argv, routines, section_names );
        return 0;
    }
    const char* routine = argv[1];
    libtest::test_func_ptr test_routine = find_tester( routine, routines );
    if (test_routine == nullptr) {
        fprintf( stderr, "%s%sError: routine %s not found%s\n\n",
                 libtest::ansi_bold, libtest::ansi_red, routine,
                 libtest::ansi_normal );
        usage( argc, argv, routines, section_names );
        return -1;
    }

    // mark fields that are used (run=false)
    Params params;
    test_routine( params, false );

    // parse parameters after routine name
    params.parse( routine, argc-2, argv+2 );

    // print input so running `test [input] > out.txt` documents input
    printf( "input: %s", argv[0] );
    for (int i = 1; i < argc; ++i) {
        printf( " %s", argv[i] );
    }
    printf( "\n" );

    // run tests
    int status = 0;
    int repeat = params.repeat.value();
    libtest::DataType last = params.datatype.value();
    params.header();
    do {
        if (params.datatype.value() != last) {
            last = params.datatype.value();
            printf( "\n" );
        }
        for (int iter = 0; iter < repeat; ++iter) {
            try {
                test_routine( params, true );
            }
            catch (blas::Error& err) {
                params.okay.value() = false;
                printf( "BLAS error: %s\n", err.what() );
            }
            catch (...) {
                // happens for assert_throw failures
                params.okay.value() = false;
                printf( "Caught error\n" );
            }
            params.print();
            status += ! params.okay.value();
        }
        if (repeat > 1) {
            printf( "\n" );
        }
    } while( params.next() );

    if (status) {
        printf( "Some tests FAILED.\n" );
    }
    else {
        printf( "All tests passed.\n" );
    }
    return status;
}
