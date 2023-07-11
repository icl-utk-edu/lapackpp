// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TEST_HH
#define TEST_HH

#include "testsweeper.hh"
#include "lapack/util.hh"
#include "matrix_params.hh"
#include "matrix_generator.hh"

// -----------------------------------------------------------------------------
using llong = long long;

// -----------------------------------------------------------------------------
class Params: public testsweeper::ParamsBase
{
public:
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double pi  = 3.141592653589793;
    const double e   = 2.718281828459045;

    Params();

    void get_range(
        int64_t n, lapack::Range* range,
        double* vl, double* vu,
        int64_t* il, int64_t* iu );

    void get_range(
        int64_t n, lapack::Range* range,
        float* vl, float* vu,
        int64_t* il, int64_t* iu );

    // Field members are explicitly public.
    // Order here determines output order.

    // ----- test matrix parameters
    MatrixParams matrix;
    MatrixParams matrixB;

    // ----- test framework parameters
    testsweeper::ParamChar   check;
    testsweeper::ParamChar   error_exit;
    testsweeper::ParamChar   ref;
    testsweeper::ParamDouble tol;
    testsweeper::ParamInt    repeat;
    testsweeper::ParamInt    verbose;
    testsweeper::ParamInt    cache;

    // ----- routine parameters
    testsweeper::ParamEnum< testsweeper::DataType > datatype;
    testsweeper::ParamEnum< blas::Layout >      layout;
    testsweeper::ParamEnum< lapack::Side >      side;
    testsweeper::ParamInt                       itype;
    testsweeper::ParamEnum< lapack::Uplo >      uplo;
    testsweeper::ParamEnum< lapack::Op >        trans;
    testsweeper::ParamEnum< lapack::Op >        transA;
    testsweeper::ParamEnum< lapack::Op >        transB;
    testsweeper::ParamEnum< lapack::Diag >      diag;
    testsweeper::ParamEnum< lapack::Norm >      norm;
    testsweeper::ParamEnum< lapack::Direction > direction;
    testsweeper::ParamEnum< lapack::StoreV >    storev;
    testsweeper::ParamInt                       ijob;   // tgsen
    testsweeper::ParamEnum< lapack::Job >       jobz;   // heev
    testsweeper::ParamEnum< lapack::Job >       jobvl;  // geev
    testsweeper::ParamEnum< lapack::Job >       jobvr;  // geev
    testsweeper::ParamEnum< lapack::Job >       jobu;   // gesvd, gesdd
    testsweeper::ParamEnum< lapack::Job >       jobvt;  // gesvd
    testsweeper::ParamEnum< lapack::Range >     range;
    testsweeper::ParamEnum< lapack::MatrixType > matrixtype;
    testsweeper::ParamEnum< lapack::Factored >  factored;
    testsweeper::ParamEnum< lapack::Equed >     equed;

    testsweeper::ParamInt3   dim;
    testsweeper::ParamInt    i;
    testsweeper::ParamInt    l;
    testsweeper::ParamInt    ka;
    testsweeper::ParamInt    kb;
    testsweeper::ParamInt    kd;
    testsweeper::ParamInt    kl;
    testsweeper::ParamInt    ku;
    testsweeper::ParamInt    nrhs;
    testsweeper::ParamInt    nb;
    testsweeper::ParamDouble vl;
    testsweeper::ParamDouble vu;
    testsweeper::ParamInt    il;
    testsweeper::ParamInt    il_out;
    testsweeper::ParamInt    iu;
    testsweeper::ParamInt    iu_out;
    testsweeper::ParamDouble fraction_start;
    testsweeper::ParamDouble fraction;
    testsweeper::ParamDouble alpha;
    testsweeper::ParamDouble beta;
    testsweeper::ParamInt    incx;
    testsweeper::ParamInt    incy;
    testsweeper::ParamInt    align;
    testsweeper::ParamInt    device;

    // ----- output parameters
    testsweeper::ParamScientific error;
    testsweeper::ParamScientific error2;
    testsweeper::ParamScientific error3;
    testsweeper::ParamScientific error4;
    testsweeper::ParamScientific error5;
    testsweeper::ParamScientific ortho;
    testsweeper::ParamScientific ortho_U;
    testsweeper::ParamScientific ortho_V;

    testsweeper::ParamDouble     time;
    testsweeper::ParamDouble     gflops;
    testsweeper::ParamDouble     gbytes;
    testsweeper::ParamInt        iters;

    testsweeper::ParamDouble     time2;
    testsweeper::ParamDouble     gflops2;
    testsweeper::ParamDouble     gbytes2;

    testsweeper::ParamDouble     time3;
    testsweeper::ParamDouble     gflops3;
    testsweeper::ParamDouble     gbytes3;

    testsweeper::ParamDouble     time4;
    testsweeper::ParamDouble     gflops4;
    testsweeper::ParamDouble     gbytes4;

    testsweeper::ParamDouble     ref_time;
    testsweeper::ParamDouble     ref_gflops;
    testsweeper::ParamDouble     ref_gbytes;
    testsweeper::ParamInt        ref_iters;

    testsweeper::ParamOkay       okay;
    testsweeper::ParamString     msg;
};

// -----------------------------------------------------------------------------
template< typename T >
inline T roundup( T x, T y )
{
    return T( (x + y - 1) / y ) * y;
}

// -----------------------------------------------------------------------------
#ifndef assert_throw
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
#endif

// -----------------------------------------------------------------------------
// Like assert(), but throws error and is not disabled by NDEBUG.
inline
void require_( bool cond, const char* condstr, const char* file, int line )
{
    if (! cond) {
        throw blas::Error( std::string(condstr) + " failed at "
                           + file + ":" + std::to_string(line) );
    }
}

#define require( cond ) require_( (cond), #cond, __FILE__, __LINE__ )

// -----------------------------------------------------------------------------
// LAPACK
// LU, general
void test_gesv  ( Params& params, bool run );
void test_gesvx ( Params& params, bool run );
void test_getrf ( Params& params, bool run );
void test_getri ( Params& params, bool run );
void test_getrs ( Params& params, bool run );
void test_gecon ( Params& params, bool run );
void test_gerfs ( Params& params, bool run );
void test_geequ ( Params& params, bool run );

// LU, band
void test_gbsv  ( Params& params, bool run );
void test_gbsvx ( Params& params, bool run );
void test_gbtrf ( Params& params, bool run );
void test_gbtrs ( Params& params, bool run );
void test_gbcon ( Params& params, bool run );
void test_gbrfs ( Params& params, bool run );
void test_gbequ ( Params& params, bool run );

// LU, tridiagonal
void test_gtsv  ( Params& params, bool run );
void test_gtsvx ( Params& params, bool run );
void test_gttrf ( Params& params, bool run );
void test_gttrs ( Params& params, bool run );
void test_gtcon ( Params& params, bool run );
void test_gtrfs ( Params& params, bool run );
void test_gtequ ( Params& params, bool run );

// Cholesky
void test_posv  ( Params& params, bool run );
void test_posvx ( Params& params, bool run );
void test_potrf ( Params& params, bool run );
void test_potri ( Params& params, bool run );
void test_potrs ( Params& params, bool run );
void test_pocon ( Params& params, bool run );
void test_porfs ( Params& params, bool run );
void test_poequ ( Params& params, bool run );

// Cholesky, packed
void test_ppsv  ( Params& params, bool run );
void test_pptrf ( Params& params, bool run );
void test_pptrs ( Params& params, bool run );
void test_pptri ( Params& params, bool run );
void test_ppcon ( Params& params, bool run );
void test_pprfs ( Params& params, bool run );
void test_ppequ ( Params& params, bool run );

// Cholesky, band
void test_pbsv  ( Params& params, bool run );
void test_pbtrf ( Params& params, bool run );
void test_pbtrs ( Params& params, bool run );
void test_pbcon ( Params& params, bool run );
void test_pbrfs ( Params& params, bool run );
void test_pbequ ( Params& params, bool run );

// Cholesky, tridiagonal
void test_ptsv  ( Params& params, bool run );
void test_pttrf ( Params& params, bool run );
void test_pttrs ( Params& params, bool run );
void test_ptcon ( Params& params, bool run );
void test_ptrfs ( Params& params, bool run );

// symmetric indefinite
void test_sysv  ( Params& params, bool run );
void test_sytrf ( Params& params, bool run );
void test_sytrs ( Params& params, bool run );
void test_sytri ( Params& params, bool run );
void test_sycon ( Params& params, bool run );
void test_syrfs ( Params& params, bool run );

// symmetric indefinite, packed
void test_spsv  ( Params& params, bool run );
void test_sptrf ( Params& params, bool run );
void test_sptrs ( Params& params, bool run );
void test_sptri ( Params& params, bool run );
void test_spcon ( Params& params, bool run );
void test_sprfs ( Params& params, bool run );

// symmetric indefinite, rook pivoting
void test_sysv_rook          ( Params& params, bool run );
void test_sytrf_rook         ( Params& params, bool run );
void test_sytrs_rook         ( Params& params, bool run );
void test_sytri_rook         ( Params& params, bool run );

// symmetric indefinite, Aasen's
void test_sysv_aa            ( Params& params, bool run );
void test_sytrf_aa           ( Params& params, bool run );
void test_sytrs_aa           ( Params& params, bool run );
void test_sytri_aa           ( Params& params, bool run );

// symmetric indefinite, Aasen's 2-stage
void test_sysv_aasen_2stage  ( Params& params, bool run );
void test_sytrf_aasen_2stage ( Params& params, bool run );
void test_sytrs_aasen_2stage ( Params& params, bool run );
void test_sytri_aasen_2stage ( Params& params, bool run );

// symmetric indefinite, rook pivoting, >= lapack 3.7
void test_sysv_rk            ( Params& params, bool run );
void test_sytrf_rk           ( Params& params, bool run );
void test_sytrs_rk           ( Params& params, bool run );
void test_sytri_rk           ( Params& params, bool run );

// hermetian
void test_hesv  ( Params& params, bool run );
void test_hetrf ( Params& params, bool run );
void test_hetrs ( Params& params, bool run );
void test_hetri ( Params& params, bool run );
void test_hecon ( Params& params, bool run );
void test_herfs ( Params& params, bool run );

// hermetian, packed
void test_hpsv  ( Params& params, bool run );
void test_hptrf ( Params& params, bool run );
void test_hptrs ( Params& params, bool run );
void test_hptri ( Params& params, bool run );
void test_hpcon ( Params& params, bool run );
void test_hprfs ( Params& params, bool run );

// least squares
void test_gels  ( Params& params, bool run );
void test_gelsy ( Params& params, bool run );
void test_gelsd ( Params& params, bool run );
void test_gelss ( Params& params, bool run );
void test_getsls( Params& params, bool run );
void test_gglse ( Params& params, bool run );
void test_ggglm ( Params& params, bool run );

// QR, LQ, QL, RQ
void test_geqr  ( Params& params, bool run );
void test_geqrf ( Params& params, bool run );
void test_gelqf ( Params& params, bool run );
void test_geqlf ( Params& params, bool run );
void test_gerqf ( Params& params, bool run );
void test_gemqrt( Params& params, bool run );

void test_ggqrf ( Params& params, bool run );
void test_gglqf ( Params& params, bool run );
void test_ggqlf ( Params& params, bool run );
void test_ggrqf ( Params& params, bool run );

void test_ungqr ( Params& params, bool run );
void test_unglq ( Params& params, bool run );
void test_ungql ( Params& params, bool run );
void test_ungrq ( Params& params, bool run );

void test_orhr_col( Params& params, bool run );
void test_unhr_col( Params& params, bool run );

void test_unmqr ( Params& params, bool run );
void test_unmlq ( Params& params, bool run );
void test_unmql ( Params& params, bool run );
void test_unmrq ( Params& params, bool run );

// triangle-pentagon QR, LQ
void test_tpqrt ( Params& params, bool run );
void test_tplqt ( Params& params, bool run );

void test_tpqrt2( Params& params, bool run );
void test_tplqt2( Params& params, bool run );

void test_tpmqrt( Params& params, bool run );
void test_tpmlqt( Params& params, bool run );

void test_tprfb ( Params& params, bool run );

// symmetric eigenvalues
void test_heev  ( Params& params, bool run );
void test_heevx ( Params& params, bool run );
void test_heevd ( Params& params, bool run );
void test_heevr ( Params& params, bool run );
void test_hetrd ( Params& params, bool run );
void test_sturm ( Params& params, bool run );
void test_ungtr ( Params& params, bool run );
void test_unmtr ( Params& params, bool run );

void test_hpev  ( Params& params, bool run );
void test_hpevx ( Params& params, bool run );
void test_hpevd ( Params& params, bool run );
void test_hpevr ( Params& params, bool run );
void test_hptrd ( Params& params, bool run );
void test_upgtr ( Params& params, bool run );
void test_upmtr ( Params& params, bool run );

void test_hbev  ( Params& params, bool run );
void test_hbevx ( Params& params, bool run );
void test_hbevd ( Params& params, bool run );
void test_hbevr ( Params& params, bool run );
void test_hbtrd ( Params& params, bool run );
void test_obgtr ( Params& params, bool run );
void test_obmtr ( Params& params, bool run );

// generalized symmetric eigenvalues
void test_hegv  ( Params& params, bool run );
void test_hegvx ( Params& params, bool run );
void test_hegvd ( Params& params, bool run );
void test_hegvr ( Params& params, bool run );
void test_hegst ( Params& params, bool run );

void test_hpgv  ( Params& params, bool run );
void test_hpgvx ( Params& params, bool run );
void test_hpgvd ( Params& params, bool run );
void test_hpgvr ( Params& params, bool run );
void test_hpgst ( Params& params, bool run );

void test_hbgv  ( Params& params, bool run );
void test_hbgvx ( Params& params, bool run );
void test_hbgvd ( Params& params, bool run );
void test_hbgvr ( Params& params, bool run );
void test_hbgst ( Params& params, bool run );

// Implements the QZ method for finding the generalized eigenvalues of the matrix pair (H,T).
void test_hgeqz ( Params& params, bool run );

// nonsymmetric eigenvalues
void test_geev  ( Params& params, bool run );
void test_geevx ( Params& params, bool run );
void test_gees  ( Params& params, bool run );
void test_geesx ( Params& params, bool run );
void test_gehrd ( Params& params, bool run );
void test_unghr ( Params& params, bool run );
void test_unmhr ( Params& params, bool run );
void test_hsein ( Params& params, bool run );
void test_trevc ( Params& params, bool run );
void test_tgexc ( Params& params, bool run );
void test_tgsen ( Params& params, bool run );

// generalized nonsymmetric eigenvalues
void test_ggev  ( Params& params, bool run );
void test_ggevx ( Params& params, bool run );
void test_gges  ( Params& params, bool run );
void test_ggesx ( Params& params, bool run );

// SVD
void test_gesvd ( Params& params, bool run );
void test_gesdd ( Params& params, bool run );
void test_gesvdx( Params& params, bool run );
void test_gesvd_2stage ( Params& params, bool run );
void test_gesdd_2stage ( Params& params, bool run );
void test_gesvdx_2stage( Params& params, bool run );
void test_gejsv ( Params& params, bool run );
void test_gesvj ( Params& params, bool run );

// auxiliary
void test_lacpy ( Params& params, bool run );
void test_laed4 ( Params& params, bool run );
void test_laset ( Params& params, bool run );
void test_laswp ( Params& params, bool run );

// auxiliary - Householder
void test_larfg ( Params& params, bool run );
void test_larfgp( Params& params, bool run );
void test_larf  ( Params& params, bool run );
void test_larfx ( Params& params, bool run );
void test_larfy ( Params& params, bool run );
void test_larfb ( Params& params, bool run );
void test_larft ( Params& params, bool run );

// auxiliary - norms
void test_lange ( Params& params, bool run );
void test_lanhe ( Params& params, bool run );
void test_lansy ( Params& params, bool run );
void test_lantr ( Params& params, bool run );
void test_lanhs ( Params& params, bool run );

// auxiliary - norms - packed
void test_lanhp ( Params& params, bool run );
void test_lansp ( Params& params, bool run );
void test_lantp ( Params& params, bool run );

// auxiliary - norms - banded
void test_langb ( Params& params, bool run );
void test_lanhb ( Params& params, bool run );
void test_lansb ( Params& params, bool run );
void test_lantb ( Params& params, bool run );

// auxiliary - norms - tridiagonal
void test_langt ( Params& params, bool run );
void test_lanht ( Params& params, bool run );
void test_lanst ( Params& params, bool run );

// auxiliary - matrix generation
void test_lagge ( Params& params, bool run );
void test_lagsy ( Params& params, bool run );
void test_laghe ( Params& params, bool run );
void test_lagtr ( Params& params, bool run );

// additional BLAS
void test_syr   ( Params& params, bool run );
void test_symv  ( Params& params, bool run );

//----------------------------------------
// GPU device functions
void test_potrf_device ( Params& params, bool run );
void test_getrf_device ( Params& params, bool run );
void test_geqrf_device ( Params& params, bool run );

#endif  //  #ifndef TEST_HH
