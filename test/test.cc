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
    blas_section,
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
   "additional BLAS",
};

// { "", nullptr, Section::newline } entries force newline in help
std::vector< libtest::routines_t > routines = {
    // -----
    // LU
    { "gesv",               test_gesv,      Section::gesv },
  //{ "gbsv",               test_gbsv,      Section::gesv },
  //{ "gtsv",               test_gtsv,      Section::gesv },
    { "",                   nullptr,        Section::newline },

  //{ "gesvx",              test_gesvx,     Section::gesv },
  //{ "gbsvx",              test_gbsvx,     Section::gesv },
  //{ "gtsvx",              test_gtsvx,     Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "getrf",              test_getrf,     Section::gesv },
  //{ "gbtrf",              test_gbtrf,     Section::gesv },
  //{ "gttrf",              test_gttrf,     Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "getrs",              test_getrs,     Section::gesv },
  //{ "gbtrs",              test_gbtrs,     Section::gesv },
  //{ "gttrs",              test_gttrs,     Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "getri",              test_getri,     Section::gesv },    // lawn 41 test
    { "",                   nullptr,        Section::newline },

    { "gecon",              test_gecon,     Section::gesv },
  //{ "gbcon",              test_gbcon,     Section::gesv },
  //{ "gtcon",              test_gtcon,     Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "gerfs",              test_gerfs,     Section::gesv },
  //{ "gbrfs",              test_gbrfs,     Section::gesv },
  //{ "gtrfs",              test_gtrfs,     Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "geequ",              test_geequ,     Section::gesv },
  //{ "gbequ",              test_gbequ,     Section::gesv },
  //{ "gtequ",              test_gtequ,     Section::gesv },
    { "",                   nullptr,        Section::newline },

    // -----
    // Cholesky
    { "posv",               test_posv,      Section::posv },
  //{ "ppsv",               test_ppsv,      Section::posv },
  //{ "pbsv",               test_pbsv,      Section::posv },
  //{ "ptsv",               test_ptsv,      Section::posv },
    { "",                   nullptr,        Section::newline },

    { "potrf",              test_potrf,     Section::posv },
  //{ "pptrf",              test_pptrf,     Section::posv },
  //{ "pbtrf",              test_pbtrf,     Section::posv },
  //{ "pttrf",              test_pttrf,     Section::posv },
    { "",                   nullptr,        Section::newline },

    { "potrs",              test_potrs,     Section::posv },
  //{ "pptrs",              test_pptrs,     Section::posv },
  //{ "pbtrs",              test_pbtrs,     Section::posv },
  //{ "pttrs",              test_pttrs,     Section::posv },
    { "",                   nullptr,        Section::newline },

    { "potri",              test_potri,     Section::posv },    // lawn 41 test
  //{ "pptri",              test_pptri,     Section::posv },
    { "",                   nullptr,        Section::newline },

    { "pocon",              test_pocon,     Section::posv },
  //{ "ppcon",              test_ppcon,     Section::posv },
  //{ "pbcon",              test_pbcon,     Section::posv },
  //{ "ptcon",              test_ptcon,     Section::posv },
    { "",                   nullptr,        Section::newline },

    { "porfs",              test_porfs,     Section::posv },
  //{ "pprfs",              test_pprfs,     Section::posv },
  //{ "pbrfs",              test_pbrfs,     Section::posv },
  //{ "ptrfs",              test_ptrfs,     Section::posv },
    { "",                   nullptr,        Section::newline },

    { "poequ",              test_poequ,     Section::posv },
  //{ "ppequ",              test_ppequ,     Section::posv },
  //{ "pbequ",              test_pbequ,     Section::posv },
  //{ "ptequ",              test_ptequ,     Section::posv },
    { "",                   nullptr,        Section::newline },

    // -----
    // symmetric indefinite
    { "sysv",               test_sysv,      Section::sysv },
  //{ "spsv",               test_spsv,      Section::sysv },
    { "",                   nullptr,        Section::newline },

    { "sytrf",              test_sytrf,     Section::sysv },
  //{ "sptrf",              test_sptrf,     Section::sysv },
    { "",                   nullptr,        Section::newline },

    { "sytrs",              test_sytrs,     Section::sysv },
  //{ "sptrs",              test_sptrs,     Section::sysv },
    { "",                   nullptr,        Section::newline },

    { "sytri",              test_sytri,     Section::sysv },
  //{ "sptri",              test_sptri,     Section::sysv },
    { "",                   nullptr,        Section::newline },

    { "sycon",              test_sycon,     Section::sysv },
  //{ "spcon",              test_spcon,     Section::sysv },
    { "",                   nullptr,        Section::newline },

    { "syrfs",              test_syrfs,     Section::sysv },
  //{ "sprfs",              test_sprfs,     Section::sysv },
    { "",                   nullptr,        Section::newline },

    // -----
  //{ "sysv_rook",          test_sysv_rook,          Section::sysv2 },
  //{ "sysv_aasen",         test_sysv_aasen,         Section::sysv2 },
  //{ "sysv_aasen_2stage",  test_sysv_aasen_2stage,  Section::sysv2 },
    { "",                   nullptr,                 Section::newline },

  //{ "sytrf_rook",         test_sytrf_rook,         Section::sysv2 },
  //{ "sytrf_aasen",        test_sytrf_aasen,        Section::sysv2 },
  //{ "sytrf_aasen_2stage", test_sytrf_aasen_2stage, Section::sysv2 },
    { "",                   nullptr,                 Section::newline },

  //{ "sytrs_rook",         test_sytrs_rook,         Section::sysv2 },
  //{ "sytrs_aasen",        test_sytrs_aasen,        Section::sysv2 },
  //{ "sytrs_aasen_2stage", test_sytrs_aasen_2stage, Section::sysv2 },
    { "",                   nullptr,                 Section::newline },

  //{ "sytri_rook",         test_sytri_rook,         Section::sysv2 },
  //{ "sytri_aasen",        test_sytri_aasen,        Section::sysv2 },
  //{ "sytri_aasen_2stage", test_sytri_aasen_2stage, Section::sysv2 },
    { "",                   nullptr,                 Section::newline },

    // -----
    // least squares
  //{ "gels",               test_gels,      Section::gels },
  //{ "gelsy",              test_gelsy,     Section::gels },
  //{ "gelsd",              test_gelsd,     Section::gels },
  //{ "gelss",              test_gelss,     Section::gels },
  //{ "getsls",             test_getsls,    Section::gels },
    { "",                   nullptr,        Section::newline },

  //{ "gglse",              test_gglse,     Section::gels },
  //{ "ggglm",              test_ggglm,     Section::gels },
    { "",                   nullptr,        Section::newline },

    // -----
    // QR, LQ, RQ, QL
  //{ "geqrf",              test_geqrf,     Section::qr },
  //{ "gelqf",              test_gelqf,     Section::qr },
  //{ "geqlf",              test_geqlf,     Section::qr },
  //{ "gerqf",              test_gerqf,     Section::qr },
    { "",                   nullptr,        Section::newline },

  //{ "ggqrf",              test_ggqrf,     Section::qr },
  //{ "gglqf",              test_gglqf,     Section::qr },
  //{ "ggqlf",              test_ggqlf,     Section::qr },
  //{ "ggrqf",              test_ggrqf,     Section::qr },
    { "",                   nullptr,        Section::newline },

  //{ "ungqr",              test_ungqr,     Section::qr },
  //{ "unglq",              test_unglq,     Section::qr },
  //{ "ungql",              test_ungql,     Section::qr },
  //{ "ungrq",              test_ungrq,     Section::qr },
    { "",                   nullptr,        Section::newline },

  //{ "unmqr",              test_unmqr,     Section::qr },
  //{ "unmlq",              test_unmlq,     Section::qr },
  //{ "unmql",              test_unmql,     Section::qr },
  //{ "unmrq",              test_unmrq,     Section::qr },
    { "",                   nullptr,        Section::newline },

    // -----
    // symmetric/Hermitian eigenvalues
  //{ "syev",               test_syev,      Section::syev },
  //{ "spev",               test_spev,      Section::syev },
  //{ "sbev",               test_sbev,      Section::syev },
    { "",                   nullptr,        Section::newline },

  //{ "syevx",              test_syevx,     Section::syev },
  //{ "spevx",              test_spevx,     Section::syev },
  //{ "sbevx",              test_sbevx,     Section::syev },
    { "",                   nullptr,        Section::newline },

  //{ "syevd",              test_syevd,     Section::syev },
  //{ "spevd",              test_spevd,     Section::syev },
  //{ "sbevd",              test_sbevd,     Section::syev },
    { "",                   nullptr,        Section::newline },

  //{ "syevr",              test_syevr,     Section::syev },
  //{ "spevr",              test_spevr,     Section::syev },
  //{ "sbevr",              test_sbevr,     Section::syev },
    { "",                   nullptr,        Section::newline },

  //{ "sytrd",              test_sytrd,     Section::syev },
  //{ "sptrd",              test_sptrd,     Section::syev },
  //{ "sbtrd",              test_sbtrd,     Section::syev },
    { "",                   nullptr,        Section::newline },

  //{ "orgtr",              test_orgtr,     Section::syev },
  //{ "opgtr",              test_opgtr,     Section::syev },
  //{ "obgtr",              test_obgtr,     Section::syev },
    { "",                   nullptr,        Section::newline },

  //{ "ormtr",              test_symtr,     Section::syev },
  //{ "opmtr",              test_spmtr,     Section::syev },
  //{ "obmtr",              test_sbmtr,     Section::syev },
    { "",                   nullptr,        Section::newline },

    // -----
    // generalized symmetric eigenvalues
  //{ "sygv",               test_sygv,      Section::sygv },
  //{ "spgv",               test_spgv,      Section::sygv },
  //{ "sbgv",               test_sbgv,      Section::sygv },
    { "",                   nullptr,        Section::newline },

  //{ "sygvx",              test_sygvx,     Section::sygv },
  //{ "spgvx",              test_spgvx,     Section::sygv },
  //{ "sbgvx",              test_sbgvx,     Section::sygv },
    { "",                   nullptr,        Section::newline },

  //{ "sygvd",              test_sygvd,     Section::sygv },
  //{ "spgvd",              test_spgvd,     Section::sygv },
  //{ "sbgvd",              test_sbgvd,     Section::sygv },
    { "",                   nullptr,        Section::newline },

  //{ "sygvr",              test_sygvr,     Section::sygv },
  //{ "spgvr",              test_spgvr,     Section::sygv },
  //{ "sbgvr",              test_sbgvr,     Section::sygv },
    { "",                   nullptr,        Section::newline },

  //{ "sygst",              test_sygst,     Section::sygv },
  //{ "spgst",              test_spgst,     Section::sygv },
  //{ "sbgst",              test_sbgst,     Section::sygv },
    { "",                   nullptr,        Section::newline },

    // -----
    // non-symmetric eigenvalues
  //{ "geev",               test_geev,      Section::geev },
  //{ "ggev",               test_ggev,      Section::geev },
    { "",                   nullptr,        Section::newline },

  //{ "geevx",              test_geevx,     Section::geev },
  //{ "ggevx",              test_ggevx,     Section::geev },
    { "",                   nullptr,        Section::newline },

  //{ "gees",               test_gees,      Section::geev },
  //{ "gges",               test_gges,      Section::geev },
    { "",                   nullptr,        Section::newline },

  //{ "geesx",              test_geesx,     Section::geev },
  //{ "ggesx",              test_ggesx,     Section::geev },
    { "",                   nullptr,        Section::newline },

  //{ "gehrd",              test_gehrd,     Section::geev },
  //{ "orghr",              test_orghr,     Section::geev },
  //{ "ormhr",              test_ormhr,     Section::geev },
  //{ "hsein",              test_hsein,     Section::geev },
  //{ "trevc",              test_trevc,     Section::geev },
    { "",                   nullptr,        Section::newline },

    // -----
    // driver: singular value decomposition
  //{ "gesvd",              test_gesvd,         Section::svd },
  //{ "gesvd_2stage",       test_gesvd_2stage,  Section::svd },
    { "",                   nullptr,            Section::newline },

  //{ "gesdd",              test_gesdd,         Section::svd },
  //{ "gesdd_2stage",       test_gesdd_2stage,  Section::svd },
    { "",                   nullptr,            Section::newline },

  //{ "gesvdx",             test_gesvdx,        Section::svd },
  //{ "gesvdx_2stage",      test_gesvdx_2stage, Section::svd },
    { "",                   nullptr,            Section::newline },

  //{ "gejsv",              test_gejsv,     Section::svd },
  //{ "gesvj",              test_gesvj,     Section::svd },
    { "",                   nullptr,        Section::newline },

    // -----
    // auxiliary
    { "lacpy",              test_lacpy,     Section::aux },
    { "laset",              test_laset,     Section::aux },
    { "laswp",              test_laswp,     Section::aux },
    { "",                   nullptr,        Section::newline },

    // auxiliary: Householder
    { "larfg",              test_larfg,     Section::aux_householder },
    { "larf",               test_larf,      Section::aux_householder },
    { "larfx",              test_larfx,     Section::aux_householder },
    { "larfb",              test_larfb,     Section::aux_householder },
    { "larft",              test_larft,     Section::aux_householder },
    { "",                   nullptr,        Section::newline },

    // auxiliary: norms
    { "lange",              test_lange,     Section::aux_norm },
    { "lanhe",              test_lanhe,     Section::aux_norm },
    { "lansy",              test_lansy,     Section::aux_norm },
    { "lantr",              test_lantr,     Section::aux_norm },
    { "",                   nullptr,        Section::newline },

    // auxiliary: matrix generation
  //{ "lagge",              test_lagge,     Section::aux_gen },
  //{ "lagsy",              test_lagsy,     Section::aux_gen },
  //{ "laghe",              test_laghe,     Section::aux_gen },
  //{ "lagtr",              test_lagtr,     Section::aux_gen },
    { "",                   nullptr,        Section::newline },

    // additional BLAS
    { "syr",                test_syr,       Section::blas_section },
    { "",                   nullptr,        Section::newline },
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
    layout    ( "layout",  6,    ParamType::List, blas::Layout::ColMajor, blas::char2layout, blas::layout2char, blas::layout2str, "layout: r=row major, c=column major" ),
    side      ( "side",    6,    ParamType::List, blas::Side::Left,       blas::char2side,   blas::side2char,   blas::side2str,   "side: l=left, r=right" ),
    uplo      ( "uplo",    6,    ParamType::List, blas::Uplo::Lower,      blas::char2uplo,   blas::uplo2char,   blas::uplo2str,   "triangle: l=lower, u=upper" ),
    trans     ( "trans",   7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose: n=no-trans, t=trans, c=conj-trans" ),
    transA    ( "transA",  7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose of A: n=no-trans, t=trans, c=conj-trans" ),
    transB    ( "transB",  7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose of B: n=no-trans, t=trans, c=conj-trans" ),
    diag      ( "diag",    7,    ParamType::List, blas::Diag::NonUnit,    blas::char2diag,   blas::diag2char,   blas::diag2str,   "diagonal: n=non-unit, u=unit" ),
    norm      ( "norm",    7,    ParamType::List, lapack::Norm::One,      lapack::char2norm, lapack::norm2char, lapack::norm2str, "norm: o=one, 2=two, i=inf, f=fro, m=max" ),
    direct    ( "direct",  8,    ParamType::List, lapack::Direct::Forward, lapack::char2direct, lapack::direct2char, lapack::direct2str, "direction: f=forward, b=backward" ),
    storev    ( "storev", 10,    ParamType::List, lapack::StoreV::Columnwise, lapack::char2storev, lapack::storev2char, lapack::storev2str, "store vectors: c=columnwise, r=rowwise" ),
    matrixtype( "matrixtype", 10, ParamType::List, lapack::MatrixType::General,
                lapack::char2matrixtype, lapack::matrixtype2char, lapack::matrixtype2str,
                "matrix type: g=general, l=lower, u=upper, h=Hessenberg, z=band-general, b=band-lower, q=band-upper" ),

    //          name,      w, p, type,            def,   min,     max, help
    dim       ( "dim",     6,    ParamType::List,          0, 1000000, "m x n x k dimensions" ),
    nrhs      ( "nrhs",    6,    ParamType::List,  10,     0, 1000000, "number of right hand sides" ),
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
        printf( "%d tests FAILED.\n", status );
    }
    else {
        printf( "All tests passed.\n" );
    }
    return status;
}
