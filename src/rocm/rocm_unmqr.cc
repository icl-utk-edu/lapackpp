// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack/defines.h"

#if defined(LAPACK_HAVE_ROCBLAS)

#include "rocm_common.hh"

#include <limits>

//==============================================================================
// todo: put into BLAS++ header somewhere.

namespace blas {
namespace internal {

rocblas_operation op2rocblas( blas::Op trans );
rocblas_side side2rocblas( blas::Side side );

} // namespace internal
} // namespace blas

//==============================================================================
namespace lapack {

//------------------------------------------------------------------------------
/// Convert int64_t to rocblas_int for rocSolver.
/// Throws if x doesn't fit in rocblas_int.
static rocblas_int to_rocblas_int_( int64_t x, const char* x_str )
{
    lapack_error_if_msg( x < std::numeric_limits<rocblas_int>::min()
                         || x > std::numeric_limits<rocblas_int>::max(),
                         "%s", x_str );
    return rocblas_int( x );
}

#define to_rocblas_int( x ) to_rocblas_int_( x, #x )

//------------------------------------------------------------------------------
// Wrapper around rocSolver workspace query.
// rocSolver ormqr and unmqr take no workspace, so the sizes are 0.
// dA is only for templating scalar_t; it isn't referenced.
template <typename scalar_t>
void unmqr_work_size_bytes(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k,
    scalar_t const* dA, int64_t ldda, scalar_t const* dtau,
    scalar_t* dC, int64_t lddc,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    *dev_work_size  = 0;
    *host_work_size = 0;
}

//------------------------------------------------------------------------------
// Intermediate wrappers around rocSolver to deal with precisions.
// Real uses ormqr, complex uses unmqr.
// rocSolver takes non-const A and tau: it modifies them during the call
// and restores them on exit.
rocblas_status rocsolver_unmqr(
    rocblas_handle solver, rocblas_side side, rocblas_operation trans,
    rocblas_int m, rocblas_int n, rocblas_int k,
    float const* dA, rocblas_int ldda, float const* dtau,
    float* dC, rocblas_int lddc )
{
    return rocsolver_sormqr(
        solver, side, trans, m, n, k,
        (float*) dA, ldda, (float*) dtau, dC, lddc );
}

//----------
rocblas_status rocsolver_unmqr(
    rocblas_handle solver, rocblas_side side, rocblas_operation trans,
    rocblas_int m, rocblas_int n, rocblas_int k,
    double const* dA, rocblas_int ldda, double const* dtau,
    double* dC, rocblas_int lddc )
{
    return rocsolver_dormqr(
        solver, side, trans, m, n, k,
        (double*) dA, ldda, (double*) dtau, dC, lddc );
}

//----------
rocblas_status rocsolver_unmqr(
    rocblas_handle solver, rocblas_side side, rocblas_operation trans,
    rocblas_int m, rocblas_int n, rocblas_int k,
    std::complex<float> const* dA, rocblas_int ldda,
    std::complex<float> const* dtau,
    std::complex<float>* dC, rocblas_int lddc )
{
    return rocsolver_cunmqr(
        solver, side, trans, m, n, k,
        (rocblas_float_complex*) dA, ldda,
        (rocblas_float_complex*) dtau,
        (rocblas_float_complex*) dC, lddc );
}

//----------
rocblas_status rocsolver_unmqr(
    rocblas_handle solver, rocblas_side side, rocblas_operation trans,
    rocblas_int m, rocblas_int n, rocblas_int k,
    std::complex<double> const* dA, rocblas_int ldda,
    std::complex<double> const* dtau,
    std::complex<double>* dC, rocblas_int lddc )
{
    return rocsolver_zunmqr(
        solver, side, trans, m, n, k,
        (rocblas_double_complex*) dA, ldda,
        (rocblas_double_complex*) dtau,
        (rocblas_double_complex*) dC, lddc );
}

//------------------------------------------------------------------------------
// Wrapper around rocSolver.
// This is async. Once finished, the return info is in dev_info on the device.
// In rocSolver, the workspaces are ignored.
// In rocSolver, there is no info; it is set to 0.
template <typename scalar_t>
void unmqr(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k,
    scalar_t const* dA, int64_t ldda, scalar_t const* dtau,
    scalar_t* dC, int64_t lddc,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue )
{
    // for real, map ConjTrans to Trans; complex doesn't take Trans.
    if (! blas::is_complex_v<scalar_t> && trans == Op::ConjTrans)
        trans = Op::Trans;
    lapack_error_if( blas::is_complex_v<scalar_t> && trans == Op::Trans );

    rocblas_int m_    = to_rocblas_int( m );
    rocblas_int n_    = to_rocblas_int( n );
    rocblas_int k_    = to_rocblas_int( k );
    rocblas_int ldda_ = to_rocblas_int( ldda );
    rocblas_int lddc_ = to_rocblas_int( lddc );

    auto solver = queue.handle();

    // for cuda, rocm, call set_device; for oneapi, do nothing.
    blas::internal_set_device( queue.device() );

    blas_dev_call(
        rocsolver_unmqr(
            solver,
            blas::internal::side2rocblas( side ),
            blas::internal::op2rocblas( trans ),
            m_, n_, k_, dA, ldda_, dtau, dC, lddc_ ));
    blas::device_memset( dev_info, 0, 1, queue );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void unmqr_work_size_bytes(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k,
    float const* dA, int64_t ldda, float const* dtau,
    float* dC, int64_t lddc,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template
void unmqr_work_size_bytes(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k,
    double const* dA, int64_t ldda, double const* dtau,
    double* dC, int64_t lddc,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template
void unmqr_work_size_bytes(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k,
    std::complex<float> const* dA, int64_t ldda,
    std::complex<float> const* dtau,
    std::complex<float>* dC, int64_t lddc,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template
void unmqr_work_size_bytes(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k,
    std::complex<double> const* dA, int64_t ldda,
    std::complex<double> const* dtau,
    std::complex<double>* dC, int64_t lddc,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

//--------------------
template
void unmqr(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k,
    float const* dA, int64_t ldda, float const* dtau,
    float* dC, int64_t lddc,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

template
void unmqr(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k,
    double const* dA, int64_t ldda, double const* dtau,
    double* dC, int64_t lddc,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

template
void unmqr(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k,
    std::complex<float> const* dA, int64_t ldda,
    std::complex<float> const* dtau,
    std::complex<float>* dC, int64_t lddc,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

template
void unmqr(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k,
    std::complex<double> const* dA, int64_t ldda,
    std::complex<double> const* dtau,
    std::complex<double>* dC, int64_t lddc,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

} // namespace lapack

#endif // LAPACK_HAVE_ROCBLAS
