// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <complex>

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "test.hh"

// -----------------------------------------------------------------------------
using testsweeper::ParamType;
using testsweeper::DataType;
using testsweeper::DataType_help;

using testsweeper::ansi_bold;
using testsweeper::ansi_red;
using testsweeper::ansi_normal;

using blas::Layout, blas::Layout_help;
using blas::Side,   blas::Side_help;
using blas::Uplo,   blas::Uplo_help;
using blas::Op,     blas::Op_help;
using blas::Diag,   blas::Diag_help;

using lapack::itype_help;
using lapack::Job;
using lapack::Job_eig_help;
using lapack::Job_eig_left_help;
using lapack::Job_eig_right_help;
using lapack::Job_svd_left_help;
using lapack::Job_svd_right_help;
using lapack::Range,      lapack::Range_help;
using lapack::Norm,       lapack::Norm_help;
using lapack::MatrixType, lapack::MatrixType_help;
using lapack::Factored,   lapack::Factored_help;
using lapack::Pivot,      lapack::Pivot_help;
using lapack::Direction,  lapack::Direction_help;
using lapack::StoreV,     lapack::StoreV_help;
using lapack::Equed,      lapack::Equed_help;

const ParamType PT_Value = ParamType::Value;
const ParamType PT_List  = ParamType::List;
const ParamType PT_Out   = ParamType::Output;

const double no_data = testsweeper::no_data_flag;
const char*  pi_rt2i = "3.141592653589793 + 1.414213562373095i";
const char*  e_rt3i  = "2.718281828459045 + 1.732050807568877i";
const double pi      = 3.141592653589793;
const double e       = 2.718281828459045;

// -----------------------------------------------------------------------------
// each section must have a corresponding entry in section_names
enum Section {
    newline = 0,  // zero flag forces newline
    gesv,
    posv,
    sysv,
    hesv,
    gels,
    qr,
    heev,
    sygv,
    geev,
    svd,
    aux,
    aux_norm,
    aux_householder,
    aux_givens,
    aux_gen,
    gpu,
    num_sections,  // last
};

const char* section_names[] = {
    "",  // none
    "LU",
    "Cholesky",
    "symmetric indefinite",
    "Hermitian indefinite",
    "least squares",
    "QR, LQ, QL, RQ",
    "symmetric eigenvalues",
    "generalized symmetric eigenvalues",
    "non-symmetric eigenvalues",
    "singular value decomposition (SVD)",
    "auxiliary",
    "matrix norms",
    "auxiliary - Householder",
    "auxiliary - Givens",
    "auxiliary - matrix generation",
    "GPU device functions",
};

// { "", nullptr, Section::newline } entries force newline in help
std::vector< testsweeper::routines_t > routines = {
    // -----
    // LU
    { "gesv",               test_gesv,      Section::gesv },
    { "gbsv",               test_gbsv,      Section::gesv },
    { "gtsv",               test_gtsv,      Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "gesvx",              test_gesvx,     Section::gesv }, // TODO Set up fact equed, (work array)=(LAPACKE rpivot)
    //{ "gbsvx",              test_gbsvx,     Section::gesv },
    //{ "gtsvx",              test_gtsvx,     Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "getrf",              test_getrf,     Section::gesv },
    { "gbtrf",              test_gbtrf,     Section::gesv },
    { "gttrf",              test_gttrf,     Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "getrs",              test_getrs,     Section::gesv },
    { "gbtrs",              test_gbtrs,     Section::gesv },
    { "gttrs",              test_gttrs,     Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "getri",              test_getri,     Section::gesv },    // lawn 41 test
    { "",                   nullptr,        Section::newline },

    { "gecon",              test_gecon,     Section::gesv },
    { "gbcon",              test_gbcon,     Section::gesv },
    { "gtcon",              test_gtcon,     Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "gerfs",              test_gerfs,     Section::gesv },
    { "gbrfs",              test_gbrfs,     Section::gesv },
    { "gtrfs",              test_gtrfs,     Section::gesv },
    { "",                   nullptr,        Section::newline },

    { "geequ",              test_geequ,     Section::gesv },
    { "gbequ",              test_gbequ,     Section::gesv },
    { "",                   nullptr,        Section::newline },

    // -----
    // Cholesky
    { "posv",               test_posv,      Section::posv },
    { "ppsv",               test_ppsv,      Section::posv },
    { "pbsv",               test_pbsv,      Section::posv },
    { "ptsv",               test_ptsv,      Section::posv },
    { "",                   nullptr,        Section::newline },

    { "potrf",              test_potrf,     Section::posv },
    { "pptrf",              test_pptrf,     Section::posv },
    { "pbtrf",              test_pbtrf,     Section::posv },
    { "pttrf",              test_pttrf,     Section::posv },
    { "",                   nullptr,        Section::newline },

    { "potrs",              test_potrs,     Section::posv },
    { "pptrs",              test_pptrs,     Section::posv },
    { "pbtrs",              test_pbtrs,     Section::posv },
    { "pttrs",              test_pttrs,     Section::posv },
    { "",                   nullptr,        Section::newline },

    { "potri",              test_potri,     Section::posv },    // lawn 41 test
    { "pptri",              test_pptri,     Section::posv },
    { "",                   nullptr,        Section::newline },

    { "pocon",              test_pocon,     Section::posv },
    { "ppcon",              test_ppcon,     Section::posv },
    { "pbcon",              test_pbcon,     Section::posv },
    { "ptcon",              test_ptcon,     Section::posv },
    { "",                   nullptr,        Section::newline },

    { "porfs",              test_porfs,     Section::posv },
    { "pprfs",              test_pprfs,     Section::posv },
    { "pbrfs",              test_pbrfs,     Section::posv },
    { "ptrfs",              test_ptrfs,     Section::posv },
    { "",                   nullptr,        Section::newline },

    { "poequ",              test_poequ,     Section::posv },
    { "ppequ",              test_ppequ,     Section::posv },
    { "pbequ",              test_pbequ,     Section::posv },
    { "",                   nullptr,        Section::newline },

    // -----
    // symmetric indefinite
    { "sysv",               test_sysv,      Section::sysv }, // tested via LAPACKE
    { "spsv",               test_spsv,      Section::sysv }, // tested via LAPACKE
    { "",                   nullptr,        Section::newline },

    { "sytrf",              test_sytrf,     Section::sysv }, // tested via LAPACKE
    { "sptrf",              test_sptrf,     Section::sysv }, // tested via LAPACKE
    { "",                   nullptr,        Section::newline },

    { "sytrs",              test_sytrs,     Section::sysv }, // tested via LAPACKE
    { "sptrs",              test_sptrs,     Section::sysv }, // tested via LAPACKE
    { "",                   nullptr,        Section::newline },

    { "sytri",              test_sytri,     Section::sysv }, // tested via LAPACKE
    { "sptri",              test_sptri,     Section::sysv }, // tested via LAPACKE
    { "",                   nullptr,        Section::newline },

    { "sycon",              test_sycon,     Section::sysv }, // tested via LAPACKE
    { "spcon",              test_spcon,     Section::sysv }, // tested via LAPACKE
    { "",                   nullptr,        Section::newline },

    { "syrfs",              test_syrfs,     Section::sysv }, // tested via LAPACKE
    { "sprfs",              test_sprfs,     Section::sysv }, // tested via LAPACKE
    { "",                   nullptr,        Section::newline },

    // -----   requires LAPACK >= 3.5
    { "sysv_rook",          test_sysv_rook,          Section::sysv }, // tested via LAPACKE using gcc/MKL
    { "sysv_rk",            test_sysv_rk,            Section::sysv }, // tested via LAPACKE using gcc/MKL
    { "sysv_aa",            test_sysv_aa,            Section::sysv }, // tested via LAPACKE using gcc/MKL
    //{ "sysv_aa_2stage",     test_sysv_aa_2stage,     Section::sysv }, // TODO No automagic generation.  No src. New call.
    { "",                   nullptr,                 Section::newline },

    { "sytrf_rook",         test_sytrf_rook,         Section::sysv }, // tested via LAPACKE using gcc/MKL
    { "sytrf_rk",           test_sytrf_rk,           Section::sysv }, // tested via LAPACKE using gcc/MKL
    { "sytrf_aa",           test_sytrf_aa,           Section::sysv }, // TODO LAPACKE wrapper broken/bugreport. Call LAPACK. Passes.
    //{ "sytrf_aa_2stage",    test_sytrf_aa_2stage,    Section::sysv }, // TODO No automagic generation.  No src. New call.
    { "",                   nullptr,                 Section::newline },

    { "sytrs_rook",         test_sytrs_rook,         Section::sysv }, // tested via LAPACKE using gcc/MKL
    //{ "sytrs_rk",           test_sytrs_rk,           Section::sysv }, // TODO the LAPACKE wrapper seems to be missing
    { "",                   nullptr,                 Section::sysv }, // space for sytrs_rk
    { "sytrs_aa",           test_sytrs_aa,           Section::sysv }, // tested via LAPACKE using gcc/MKL
    //{ "sytrs_aa_2stage",    test_sytrs_aa_2stage,    Section::sysv }, // TODO No automagic generation.  No src. New call.
    { "",                   nullptr,                 Section::newline },

    //{ "sytri_rook",         test_sytri_rook,         Section::sysv }, // TODO lapack_fortran.h header missing
    { "",                   nullptr,                 Section::newline },

    // -----
    // Hermitian indefinite
    { "hesv",               test_hesv,      Section::hesv }, // tested via LAPACKE
    { "hpsv",               test_hpsv,      Section::hesv }, // tested via LAPACKE
    { "",                   nullptr,        Section::newline },

    { "hetrf",              test_hetrf,     Section::hesv }, // tested via LAPACKE
    { "hptrf",              test_hptrf,     Section::hesv }, // tested via LAPACKE
    { "",                   nullptr,        Section::newline },

    { "hetrs",              test_hetrs,     Section::hesv }, // tested via LAPACKE
    { "hptrs",              test_hptrs,     Section::hesv }, // tested via LAPACKE
    { "",                   nullptr,        Section::newline },

    { "hetri",              test_hetri,     Section::hesv }, // tested via LAPACKE
    { "hptri",              test_hptri,     Section::hesv }, // tested via LAPACKE
    { "",                   nullptr,        Section::newline },

    { "hecon",              test_hecon,     Section::hesv }, // tested via LAPACKE
    { "hpcon",              test_hpcon,     Section::hesv }, // tested via LAPACKE, error < 3*eps
    { "",                   nullptr,        Section::newline },

    { "herfs",              test_herfs,     Section::hesv }, // tested via LAPACKE
    { "hprfs",              test_hprfs,     Section::hesv }, // tested via LAPACKE, error < 3*eps
    { "",                   nullptr,        Section::newline },

    // -----
    // least squares
    { "gels",               test_gels,      Section::gels }, // tested via LAPACKE using gcc/MKL
    { "gelsy",              test_gelsy,     Section::gels }, // tested via LAPACKE using gcc/MKL TODO jpvt[i]=i rcond=0
    { "gelsd",              test_gelsd,     Section::gels }, // TODO: Segfaults for some Z sizes. src/gelsd.cc:275 lrwork_ too small?
    { "gelss",              test_gelss,     Section::gels }, // tested via LAPACKE using gcc/MKL TODO rcond=n
    { "getsls",             test_getsls,    Section::gels }, // tested via LAPACKE using gcc/MKL
    { "",                   nullptr,        Section::newline },

    { "gglse",              test_gglse,     Section::gels }, // tested via LAPACKE using gcc/MKL
    { "ggglm",              test_ggglm,     Section::gels }, // tested via LAPACKE using gcc/MKL
    { "",                   nullptr,        Section::newline },

    // -----
    // QR, LQ, RQ, QL
    { "geqr",               test_geqr,      Section::qr }, // tested numerically
    { "geqrf",              test_geqrf,     Section::qr }, // tested numerically
    { "gelqf",              test_gelqf,     Section::qr }, // tested numerically
    { "geqlf",              test_geqlf,     Section::qr }, // tested numerically
    { "gerqf",              test_gerqf,     Section::qr }, // tested numerically; R, Q are full sizeof(A), could be smaller
    { "gemqrt",             test_gemqrt,    Section::qr }, // tested via LAPACKE
    { "",                   nullptr,        Section::newline },

    { "ggqrf",              test_ggqrf,     Section::qr }, // tested via LAPACKE using gcc/MKL, TODO for now use p=param.k
    //{ "gglqf",              test_gglqf,     Section::qr }, // TODO No automagic generation.  No src
    { "",                   nullptr,        Section::qr }, // space for gglqf
    //{ "ggqlf",              test_ggqlf,     Section::qr }, // TODO No automagic generation.  No src
    { "",                   nullptr,        Section::qr }, // space for ggqlf
    { "ggrqf",              test_ggrqf,     Section::qr }, // tested via LAPACKE using gcc/MKL, TODO for now use p=param.k
    { "",                   nullptr,        Section::newline },

    { "ungqr",              test_ungqr,     Section::qr }, // tested numerically based on lapack; R, Q full sizes
    { "unglq",              test_unglq,     Section::qr }, // tested numerically based on lapack; R, Q full; m<=n, k<=m
    { "ungql",              test_ungql,     Section::qr }, // tested numerically based on lapack; R, Q full sizes
    { "ungrq",              test_ungrq,     Section::qr }, // tested numerically based on lapack; R, Q full sizes
    { "",                   nullptr,        Section::newline },

    { "orhr_col",           test_orhr_col,  Section::qr },
    { "unhr_col",           test_unhr_col,  Section::qr },
    { "",                   nullptr,        Section::newline },

    //{ "unmqr",              test_unmqr,     Section::qr }, // TODO segfaults
    //{ "unmlq",              test_unmlq,     Section::qr },
    //{ "unmql",              test_unmql,     Section::qr },
    //{ "unmrq",              test_unmrq,     Section::qr },
    //{ "",                   nullptr,        Section::newline },

    { "tpqrt",              test_tpqrt,     Section::qr },
    { "tplqt",              test_tplqt,     Section::qr },
    { "",                   nullptr,        Section::newline },

    { "tpqrt2",             test_tpqrt2,    Section::qr },
    { "tplqt2",             test_tplqt2,    Section::qr },
    { "",                   nullptr,        Section::newline },

    { "tpmqrt",             test_tpmqrt,    Section::qr },
    { "tpmlqt",             test_tpmlqt,    Section::qr },
    { "",                   nullptr,        Section::newline },

    { "tprfb",              test_tprfb,     Section::qr },
    { "",                   nullptr,        Section::newline },

    // -----
    // symmetric/Hermitian eigenvalues
    { "heev",               test_heev,      Section::heev }, // backward error check
    { "hpev",               test_hpev,      Section::heev }, // tested via LAPACKE
    { "hbev",               test_hbev,      Section::heev }, // tested via LAPACKE
    { "sturm",              test_sturm,     Section::heev },
    { "",                   nullptr,        Section::newline },

    { "heevx",              test_heevx,     Section::heev }, // backward error check
    { "hpevx",              test_hpevx,     Section::heev }, // tested via LAPACKE
    { "hbevx",              test_hbevx,     Section::heev }, // tested via LAPACKE
    { "",                   nullptr,        Section::newline },

    { "heevd",              test_heevd,     Section::heev }, // backward error check
    { "hpevd",              test_hpevd,     Section::heev }, // tested via LAPACKE using gcc/MKL
    { "hbevd",              test_hbevd,     Section::heev }, // tested via LAPACKE using gcc/MKL
    { "",                   nullptr,        Section::newline },

    { "heevr",              test_heevr,     Section::heev }, // backward error check
    { "lae2",               test_lae2,      Section::heev }, // forward  error check, compared to laev2
    { "laev2",              test_laev2,     Section::heev }, // backward error check
    { "",                   nullptr,        Section::newline },

    { "hetrd",              test_hetrd,     Section::heev }, // tested via LAPACKE using gcc/MKL
    { "hptrd",              test_hptrd,     Section::heev }, // tested via LAPACKE using gcc/MKL
    //{ "hbtrd",              test_hbtrd,     Section::heev }, // Need to add to test.cc params a new vect option v,n,u for forming Q
    { "",                   nullptr,        Section::newline },

    { "ungtr",              test_ungtr,     Section::heev }, // tested via LAPACKE using gcc/MKL
    { "upgtr",              test_upgtr,     Section::heev }, // tested via LAPACKE using gcc/MKL
    { "",                   nullptr,        Section::newline },

    { "unmtr",              test_unmtr,     Section::heev }, // tested via LAPACKE using gcc/MKL
    { "upmtr",              test_upmtr,     Section::heev },
    { "",                   nullptr,        Section::newline },

    // -----
    // generalized symmetric eigenvalues
    { "hegv",               test_hegv,      Section::sygv }, // tested via LAPACKE using gcc/MKL
    { "hpgv",               test_hpgv,      Section::sygv }, // tested via LAPACKE using gcc/MKL
    { "hbgv",               test_hbgv,      Section::sygv }, // tested via LAPACKE using gcc/MKL
    { "",                   nullptr,        Section::newline },

    { "hegvx",              test_hegvx,     Section::sygv }, // tested via LAPACKE using gcc/MKL
    { "hpgvx",              test_hpgvx,     Section::sygv }, // tested via LAPACKE using gcc/MKL
    { "hbgvx",              test_hbgvx,     Section::sygv }, // tested via LAPACKE using gcc/MKL
    { "",                   nullptr,        Section::newline },

    { "hegvd",              test_hegvd,     Section::sygv }, // tested via LAPACKE using gcc/MKL
    { "hpgvd",              test_hpgvd,     Section::sygv }, // tested via LAPACKE using gcc/MKL
    { "hbgvd",              test_hbgvd,     Section::sygv }, // TODO Segfaults.. is the src correct?
    { "",                   nullptr,        Section::newline },

    { "hegst",              test_hegst,     Section::sygv }, // tested via LAPACKE using gcc/MKL
    { "hpgst",              test_hpgst,     Section::sygv }, // tested via LAPACKE using gcc/MKL
    //{ "hbgst",              test_hbgst,     Section::sygv }, // TODO This test requires non-working --vect flag

    //{ "hgeqz",              test_hgeqz,     Section::sygv }, // TODO Needs params job compq compz...
    { "",                   nullptr,        Section::newline },

    // -----
    // non-symmetric eigenvalues
    { "geev",               test_geev,      Section::geev },
    { "ggev",               test_ggev,      Section::geev }, // tested via LAPACKE using gcc/MKL. NOTE: No doxygen in src/ggev.cc
    { "",                   nullptr,        Section::newline },

    //{ "geevx",              test_geevx,     Section::geev }, // TODO No src
    //{ "ggevx",              test_ggevx,     Section::geev }, // TODO No src
    { "",                   nullptr,        Section::newline },

    //{ "gees",               test_gees,      Section::geev }, // TODO needs --sort --select, external SELECT logical function
    //{ "gges",               test_gges,      Section::geev }, // TODO needs SELCTG (external sort procedure) LOGICAL FUNCTION
    { "",                   nullptr,        Section::newline },

    //{ "geesx",              test_geesx,     Section::geev }, // TODO needs external select function
    //{ "ggesx",              test_ggesx,     Section::geev }, // TODO needs external select function
    { "",                   nullptr,        Section::newline },

    { "gehrd",              test_gehrd,     Section::geev }, // TODO Fixed ilo=1, ihi=n, should these vary?
    { "unghr",              test_unghr,     Section::geev }, // TODO Fixed ilo=1, ihi=n, should these vary?
    { "unmhr",              test_unmhr,     Section::geev },
    //{ "hsein",              test_hsein,     Section::geev }, // TODO error in automagic generation KeyError eigsrc
    //{ "trevc",              test_trevc,     Section::geev }, // TODO --howmany, need to setup a bool select array
    { "",                   nullptr,        Section::newline },

    { "tgexc",              test_tgexc,     Section::geev },
    { "tgsen",              test_tgsen,     Section::geev },
    { "",                   nullptr,        Section::newline },

    // -----
    // driver: singular value decomposition
    { "gesvd",              test_gesvd,         Section::svd },
    //{ "gesvd_2stage",       test_gesvd_2stage,  Section::svd }, // TODO No src
    { "",                   nullptr,            Section::newline },

    { "gesdd",              test_gesdd,         Section::svd },
    //{ "gesdd_2stage",       test_gesdd_2stage,  Section::svd }, // TODO No src
    { "",                   nullptr,            Section::newline },

    { "gesvdx",             test_gesvdx,        Section::svd }, // tested via LAPACKE using gcc/MKL
    //{ "gesvdx_2stage",      test_gesvdx_2stage, Section::svd }, // TODO No src
    { "",                   nullptr,            Section::newline },

    //{ "gejsv",              test_gejsv,     Section::svd }, // TODO No src
    //{ "gesvj",              test_gesvj,     Section::svd }, // TODO No src
    { "",                   nullptr,        Section::newline },

    // -----
    // auxiliary
    { "lacpy",              test_lacpy,     Section::aux },
    { "laed4",              test_laed4,     Section::aux },
    { "laset",              test_laset,     Section::aux },
    { "laswp",              test_laswp,     Section::aux },
    { "",                   nullptr,        Section::newline },

    // auxiliary: Householder
    { "larfg",              test_larfg,     Section::aux_householder },
    { "larfgp",             test_larfgp,    Section::aux_householder },
    { "larf",               test_larf,      Section::aux_householder },
    { "larfx",              test_larfx,     Section::aux_householder },
    { "larfy",              test_larfy,     Section::aux_householder },
    { "larfb",              test_larfb,     Section::aux_householder },
    { "larft",              test_larft,     Section::aux_householder },
    { "",                   nullptr,        Section::newline },

    // auxiliary: Givens rotations
    { "lasr",               test_lasr,      Section::aux_givens },  // forward error check, compared to rot
    { "",                   nullptr,        Section::newline },

    // auxiliary: norms
    { "lange",              test_lange,     Section::aux_norm },
    { "lanhe",              test_lanhe,     Section::aux_norm },
    { "lansy",              test_lansy,     Section::aux_norm },
    { "lantr",              test_lantr,     Section::aux_norm },
    { "lanhs",              test_lanhs,     Section::aux_norm },
    { "",                   nullptr,        Section::newline },

    // auxiliary: norms - packed
    { "",                   nullptr,        Section::aux_norm },
    { "lanhp",              test_lanhp,     Section::aux_norm },
    { "lansp",              test_lansp,     Section::aux_norm },
    { "lantp",              test_lantp,     Section::aux_norm },
    { "",                   nullptr,        Section::newline },

    // auxiliary: norms - banded
    { "langb",              test_langb,     Section::aux_norm },
    { "lanhb",              test_lanhb,     Section::aux_norm },
    { "lansb",              test_lansb,     Section::aux_norm },
    { "lantb",              test_lantb,     Section::aux_norm },
    { "",                   nullptr,        Section::newline },

    // auxiliary: norms - tridiagonal
    { "langt",              test_langt,     Section::aux_norm },
    { "lanht",              test_lanht,     Section::aux_norm },
    { "lanst",              test_lanst,     Section::aux_norm },
    { "",                   nullptr,        Section::newline },

    // auxiliary: matrix generation
    //{ "lagge",              test_lagge,     Section::aux_gen },
    //{ "lagsy",              test_lagsy,     Section::aux_gen },
    //{ "laghe",              test_laghe,     Section::aux_gen },
    //{ "lagtr",              test_lagtr,     Section::aux_gen },
    { "",                   nullptr,        Section::newline },

    //----------------------------------------
    // GPU device functions
    { "dev-potrf",          test_potrf_device,      Section::gpu },
    { "dev-getrf",          test_getrf_device,      Section::gpu },
    { "dev-geqrf",          test_geqrf_device,      Section::gpu },
    { "dev-heevd",          test_heevd_device,      Section::gpu },
    { "dev-larfg",          test_larfg_device,      Section::gpu },
    { "dev-tpqrt",          test_tpqrt_device,      Section::gpu },
    { "dev-tpqrt2",         test_tpqrt2_device,     Section::gpu },
    { "dev-tprfb",          test_tprfb_device,      Section::gpu },
    { "",                   nullptr,                Section::newline },
};

// -----------------------------------------------------------------------------
// Params class
// List of parameters

Params::Params():
    ParamsBase(),

    // w = width
    // p = precision
    //----- test framework parameters
    //          name,         w, type, default, valid, help
    check     ( "check",      0, PT_Value, 'y', "ny", "check the results" ),
    error_exit( "error-exit", 0, PT_Value, 'n', "ny", "check error exits" ),
    ref       ( "ref",        0, PT_Value, 'n', "ny", "run reference; sometimes check implies ref" ),

    //          name,         w, p, type, default,  min,  max, help
    tol       ( "tol",        0, 0, PT_Value,  50,    1, 1000, "tolerance (e.g., error < tol*epsilon to pass)" ),
    repeat    ( "repeat",     0,    PT_Value,   1,    1, 1000, "times to repeat each test" ),
    verbose   ( "verbose",    0,    PT_Value,   0,    0,   10, "verbose level" ),
    cache     ( "cache",      0,    PT_Value,  20,    1, 1024, "total cache size, in MiB" ),

    //----- routine parameters, enums
    //          name,         w, type,    default, help
    datatype  ( "type",       4, PT_List, DataType::Double, DataType_help ),
    // BLAS & LAPACK options
    layout    ( "layout",     6, PT_List, Layout::ColMajor, Layout_help ),
    ijob      ( "ijob",       5, PT_List, 0, 0, 5,
                "condition numbers to compute, 0 to 5; see tgsen docs" ),
    itype     ( "itype",      5, PT_List, 1, 1, 3, itype_help ),
    jobz      ( "jobz",       5, PT_List, Job::NoVec, Job_eig_help ),
    jobvl     ( "jobvl",      5, PT_List, Job::NoVec, Job_eig_left_help ),
    jobvr     ( "jobvr",      5, PT_List, Job::NoVec, Job_eig_right_help ),
    jobu      ( "jobu",       9, PT_List, Job::NoVec, Job_svd_left_help ),
    jobvt     ( "jobvt",      9, PT_List, Job::NoVec, Job_svd_right_help ),
    // range is set by vl, vu, il, iu, fraction
    range     ( "range",      9, PT_List, Range::All, Range_help ),
    norm      ( "norm",       4, PT_List, Norm::One, Norm_help ),
    matrixtype( "matrixtype", 10, PT_List, MatrixType::General, MatrixType_help ),
    factored  ( "factored",   11, PT_List, Factored::NotFactored, Factored_help ),
    side      ( "side",       6, PT_List, Side::Left, Side_help ),
    uplo      ( "uplo",       6, PT_List, Uplo::Lower, Uplo_help ),
    trans     ( "trans",      7, PT_List, Op::NoTrans, Op_help ),
    transA    ( "transA",     7, PT_List, Op::NoTrans, Op_help ),
    transB    ( "transB",     7, PT_List, Op::NoTrans, Op_help ),
    diag      ( "diag",       7, PT_List, Diag::NonUnit, Diag_help ),
    pivot     ( "pivot",      8, PT_List, Pivot::Variable, Pivot_help ),
    direction ( "direction",  8, PT_List, Direction::Forward, Direction_help ),
    storev    ( "storev",     7, PT_List, StoreV::Columnwise, StoreV_help ),
    equed     ( "equed",      5, PT_List, Equed::Both, Equed_help ),

    //----- routine parameters, numeric
    //          name,         w, p, type,    default,  min,  max, help
    dim       ( "dim",        6,    PT_List,             0, 1e10, "m by n by k dimensions" ),
    i         ( "i",          6,    PT_List,       1,    0, 1e10, "i index (e.g., laed4)" ),
    l         ( "l",          6,    PT_List,      10,    0, 1e10, "l dimension (e.g., tpqrt)" ),
    ka        ( "ka",         6,    PT_List,      10,    0,  1e6, "bandwidth of A" ),
    kb        ( "kb",         6,    PT_List,      10,    0,  1e6, "bandwidth of B" ),
    kd        ( "kd",         6,    PT_List,      10,    0,  1e6, "bandwidth" ),
    kl        ( "kl",         6,    PT_List,      10,    0,  1e6, "lower bandwidth" ),
    ku        ( "ku",         6,    PT_List,      10,    0,  1e6, "upper bandwidth" ),
    nrhs      ( "nrhs",       6,    PT_List,      10,    0, 1e10, "number of right hand sides" ),
    nb        ( "nb",         4,    PT_List,     384,    0,  1e6, "block size" ),

    vl        ( "vl",         6, 3, PT_List,    -inf, -inf,  inf, "lower bound of eigen/singular values to find" ),
    vu        ( "vu",         6, 3, PT_List,     inf, -inf,  inf, "upper bound of eigen/singular values to find" ),
    // input il, iu, or fraction; output {il, iu}_out adjusted for matrix size or set by fraction
    il        ( "il",         0,    PT_List,       1,    1, 1e10, "1-based index of smallest eigen/singular value to find" ),
    iu        ( "iu",         0,    PT_List,      -1,   -1, 1e10, "1-based index of largest  eigen/singular value to find; -1 is all" ),
    il_out    ( "il",         6,    PT_Out,        1,    1, 1e10, "1-based index of smallest eigen/singular value to find (actual value used)" ),
    iu_out    ( "iu",         6,    PT_Out,       -1,   -1, 1e10, "1-based index of largest  eigen/singular value to find (actual value used)" ),
    fraction_start( "fraction-start",
                              0, 0, PT_List,       0,    0,    1, "index of smallest eigen/singular value to find, as fraction of n; sets il = 1 + fraction_start*n" ),
    fraction  ( "fraction",   0, 0, PT_List,       1,    0,    1, "fraction of eigen/singular values to find; sets iu = il - 1 + fraction*n" ),

    alpha     ( "alpha",      3, 1, PT_List, pi_rt2i, -inf,  inf, "scalar alpha" ),
    beta      ( "beta",       3, 1, PT_List,  e_rt3i, -inf,  inf, "scalar beta" ),
    incx      ( "incx",       4,    PT_List,       1, -1e3,  1e3, "stride of x vector" ),
    incy      ( "incy",       4,    PT_List,       1, -1e3,  1e3, "stride of y vector" ),
    align     ( "align",      0,    PT_List,       1,    1, 1024, "column alignment (sets lda, ldb, etc. to multiple of align)" ),
    device    ( "device",     6,    PT_List,       0,    0,  100, "device id" ),

    //----- output parameters
    // min, max are ignored
    // error:   %8.2e allows 9.99e-99
    // time:    %9.3f allows 99999.999 s = 2.9 days
    // gflops: %12.3f allows 99999999.999 Gflop/s = 100 Pflop/s
    //          name,         w, p, type,   default, min, max, help
    error     ( "error",      8, 2, PT_Out, no_data, 0, 0, "numerical error" ),
    error2    ( "error2",     8, 2, PT_Out, no_data, 0, 0, "numerical error" ),
    error3    ( "error3",     8, 2, PT_Out, no_data, 0, 0, "numerical error" ),
    error4    ( "error4",     8, 2, PT_Out, no_data, 0, 0, "numerical error" ),
    error5    ( "error5",     8, 2, PT_Out, no_data, 0, 0, "numerical error" ),
    ortho     ( "orth.",      8, 2, PT_Out, no_data, 0, 0, "orthogonality error" ),
    ortho_U   ( "U orth.",    8, 2, PT_Out, no_data, 0, 0, "U orthogonality error" ),
    ortho_V   ( "V orth.",    8, 2, PT_Out, no_data, 0, 0, "V orthogonality error" ),

    time      ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "time to solution" ),
    gflops    ( "gflop/s",   12, 3, PT_Out, no_data, 0, 0, "Gflop/s rate" ),
    gbytes    ( "gbyte/s",   12, 3, PT_Out, no_data, 0, 0, "Gbyte/s rate" ),
    iters     ( "iters",      5,    PT_Out, 0,       0, 0, "iterations to solution" ),

    ref_time  ( "ref time (s)",  9, 3, PT_Out, no_data, 0, 0, "reference time to solution" ),
    ref_gflops( "ref gflop/s",  12, 3, PT_Out, no_data, 0, 0, "reference Gflop/s rate" ),
    ref_gbytes( "ref gbyte/s",  12, 3, PT_Out, no_data, 0, 0, "reference Gbyte/s rate" ),
    ref_iters ( "ref iters",     5,    PT_Out, 0,       0, 0, "reference iterations to solution" ),

    // default -1 means "no check"
    //          name,         w, type, default, min, max, help
    okay      ( "status",     6, PT_Out,    -1, 0, 0, "success indicator" ),
    msg       ( "",           1, PT_Out,    "",       "error message" )
{
    // change names of matrix B's params
    matrixB.kind.name( "matrixB" );
    matrixB.cond.name( "condB" );
    matrixB.condD.name( "condD_B" );

    // mark standard set of output fields as used
    okay();
    error();
    time();

    // mark framework parameters as used, so they will be accepted on the command line
    check();
    error_exit();
    ref();
    repeat();
    verbose();
    cache();

    // routine's parameters are marked by the test routine; see main
}

// -----------------------------------------------------------------------------
// determines the range, il, iu, vl, vu values.
void Params::get_range(
    int64_t n, lapack::Range* range_arg,
    double* vl_arg, double* vu_arg,
    int64_t* il_arg, int64_t* iu_arg )
{
    // default assume All
    *vl_arg = this->vl();
    *vu_arg = this->vu();
    *il_arg = std::min( this->il(), n );
    *iu_arg = std::min( this->iu(), n );
    if (*iu_arg == -1)
        *iu_arg = n;
    double frac_start = this->fraction_start();
    double frac = this->fraction();
    if (frac_start + frac > 1)
        throw lapack::Error( "Error: fraction_start + fraction > 1" );

    // set range based on fraction, il/iu, vl/vu values
    if (frac != 1) {
        *range_arg = lapack::Range::Index;
        *il_arg = std::min( 1 + int64_t( frac_start * n ), n );
        *iu_arg = std::min( (*il_arg) - 1 + int64_t( frac * n ), n );
    }
    else if (*il_arg != 1 || *iu_arg != n) {
        *range_arg = lapack::Range::Index;
    }
    else if (*vl_arg != -inf || *vu_arg != inf) {
        *range_arg = lapack::Range::Value;
    }
    else {
        *range_arg = lapack::Range::All;
    }

    this->range() = *range_arg;
    this->il_out() = *il_arg;
    this->iu_out() = *iu_arg;
}

// -----------------------------------------------------------------------------
void Params::get_range(
    int64_t n, lapack::Range* range_arg,
    float* vl_arg, float* vu_arg,
    int64_t* il_arg, int64_t* iu_arg )
{
    double dvl, dvu;
    this->get_range( n, range_arg, &dvl, &dvu, il_arg, iu_arg );
    *vl_arg = float(dvl);
    *vu_arg = float(dvu);
}

// -----------------------------------------------------------------------------
// Compare a == b, bitwise. Returns true if a and b are both the same NaN value,
// unlike (a == b) which is false for NaNs.
bool same( double a, double b );

bool same( double a, double b )
{
    return (memcmp( &a, &b, sizeof(double) ) == 0);
}

// -----------------------------------------------------------------------------
// Prints line describing matrix kind and cond, if kind or cond changed.
// Updates kind and cond to current values.
void print_matrix_header(
    MatrixParams& params, const char* caption,
    std::string* matrix, double* cond, double* condD );

void print_matrix_header(
    MatrixParams& params, const char* caption,
    std::string* matrix, double* cond, double* condD )
{
    if (params.kind.used() &&
        (*matrix != params.kind() ||
         ! same( *cond,  params.cond_used() ) ||
         ! same( *condD, params.condD() )))
    {
        *matrix = params.kind();
        *cond   = params.cond_used();
        *condD  = params.condD();
        printf( "%s: %s, cond(S) = ", caption, matrix->c_str() );
        if (std::isnan( *cond ))
            printf( "NA" );
        else
            printf( "%.2e", *cond );
        if (! std::isnan(*condD))
            printf( ", cond(D) = %.2e", *condD );
        printf( "\n" );
    }
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    using testsweeper::QuitException;

    // These may or may not be used; mark unused to silence warnings.
    blas_unused( pi_rt2i );
    blas_unused( e_rt3i  );
    blas_unused( pi      );
    blas_unused( e       );

    // check that all sections have names
    require( sizeof(section_names)/sizeof(*section_names) == Section::num_sections );

    int status = 0;
    try {
        int version = lapack::lapackpp_version();
        printf( "LAPACK++ version %d.%02d.%02d, id %s\n",
                version / 10000, (version % 10000) / 100, version % 100,
                lapack::lapackpp_id() );

        // print input so running `test [input] > out.txt` documents input
        printf( "input: %s", argv[0] );
        for (int i = 1; i < argc; ++i) {
            // quote arg if necessary
            std::string arg( argv[i] );
            const char* wordchars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-=";
            if (arg.find_first_not_of( wordchars ) != std::string::npos)
                printf( " '%s'", argv[i] );
            else
                printf( " %s", argv[i] );
        }
        printf( "\n" );

        // Usage: test [params] routine
        if (argc < 2
            || strcmp( argv[argc-1], "-h" ) == 0
            || strcmp( argv[argc-1], "--help" ) == 0)
        {
            usage( argc, argv, routines, section_names );
            throw QuitException();
        }

        if (strcmp( argv[argc-1], "--help-matrix" ) == 0) {
            lapack::generate_matrix_usage();
            throw QuitException();
        }

        // find routine to test
        const char* routine = argv[ argc-1 ];
        testsweeper::test_func_ptr test_routine = find_tester( routine, routines );
        if (test_routine == nullptr) {
            usage( argc, argv, routines, section_names );
            throw std::runtime_error(
                std::string("routine ") + routine + " not found" );
        }

        // mark fields that are used (run=false)
        Params params;
        test_routine( params, false );

        // Parse parameters up to routine name.
        try {
            params.parse( routine, argc-2, argv+1 );
        }
        catch (const std::exception& ex) {
            params.help( routine );
            throw;
        }

        // show align column if it has non-default values
        if (params.align.size() != 1 || params.align() != 1) {
            params.align.width( 5 );
        }

        // run tests
        int repeat = params.repeat();
        std::vector<double> times( repeat ), gflops( repeat );
        testsweeper::DataType last = params.datatype();
        std::string matrix, matrixB;
        double cond = 0, condD = 0, condB = 0, condD_B = 0;
        params.header();
        do {
            if (params.datatype() != last) {
                last = params.datatype();
                printf( "\n" );
            }
            for (int iter = 0; iter < repeat; ++iter) {
                try {
                    test_routine( params, true );
                }
                catch (const std::exception& ex) {
                    fprintf( stderr, "%s%sError: %s%s\n",
                             ansi_bold, ansi_red, ex.what(), ansi_normal );
                    params.okay() = false;
                }

                // Collect stats.
                times [ iter ] = params.time();
                gflops[ iter ] = params.gflops();

                if (iter == 0) {
                    print_matrix_header( params.matrix,  "test matrix A", &matrix,  &cond,  &condD   );
                    print_matrix_header( params.matrixB, "test matrix B", &matrixB, &condB, &condD_B );
                }
                params.print();
                fflush( stdout );
                status += ! params.okay();
                params.reset_output();
            }
            if (repeat > 1) {
                testsweeper::print_stats( params.time,   times  );
                testsweeper::print_stats( params.gflops, gflops );
                printf( "\n" );
            }
        } while(params.next());

        if (status) {
            printf( "%d tests FAILED for %s.\n", status, routine );
        }
        else {
            printf( "All tests passed for %s.\n", routine );
        }
    }
    catch (const QuitException& ex) {
        // pass: no error to print
    }
    catch (const std::exception& ex) {
        fprintf( stderr, "\n%s%sError: %s%s\n",
                 ansi_bold, ansi_red, ex.what(), ansi_normal );
        status = -1;
    }

    return status;
}
