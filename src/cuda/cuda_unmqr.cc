// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack/defines.h"

#if defined(LAPACK_HAVE_CUBLAS)

#include "lapack/device.hh"
#include "cuda_common.hh"

#include <limits>

//==============================================================================
// todo: put into BLAS++ header somewhere.

namespace blas {
namespace internal {

cublasOperation_t op2cublas( blas::Op trans );
cublasSideMode_t side2cublas( blas::Side side );

} // namespace internal
} // namespace blas

//==============================================================================
namespace lapack {

//------------------------------------------------------------------------------
/// Convert int64_t to int for cuSolver ormqr and unmqr, which have only a
/// 32-bit int interface (no cusolverDnX version).
/// Throws if x doesn't fit in int.
static int to_cusolver_int_( int64_t x, const char* x_str )
{
    lapack_error_if_msg( x < std::numeric_limits<int>::min()
                         || x > std::numeric_limits<int>::max(), "%s", x_str );
    return int( x );
}

#define to_cusolver_int( x ) to_cusolver_int_( x, #x )

//------------------------------------------------------------------------------
// Intermediate wrappers around cuSolver to deal with precisions.
// Real uses ormqr, complex uses unmqr.
cusolverStatus_t cusolver_unmqr_bufferSize(
    cusolverDnHandle_t solver, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k,
    float const* dA, int ldda, float const* dtau,
    float const* dC, int lddc, int* lwork )
{
    return cusolverDnSormqr_bufferSize(
        solver, side, trans, m, n, k, dA, ldda, dtau, dC, lddc, lwork );
}

//----------
cusolverStatus_t cusolver_unmqr_bufferSize(
    cusolverDnHandle_t solver, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k,
    double const* dA, int ldda, double const* dtau,
    double const* dC, int lddc, int* lwork )
{
    return cusolverDnDormqr_bufferSize(
        solver, side, trans, m, n, k, dA, ldda, dtau, dC, lddc, lwork );
}

//----------
cusolverStatus_t cusolver_unmqr_bufferSize(
    cusolverDnHandle_t solver, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k,
    std::complex<float> const* dA, int ldda, std::complex<float> const* dtau,
    std::complex<float> const* dC, int lddc, int* lwork )
{
    return cusolverDnCunmqr_bufferSize(
        solver, side, trans, m, n, k,
        (cuFloatComplex const*) dA, ldda,
        (cuFloatComplex const*) dtau,
        (cuFloatComplex const*) dC, lddc, lwork );
}

//----------
cusolverStatus_t cusolver_unmqr_bufferSize(
    cusolverDnHandle_t solver, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k,
    std::complex<double> const* dA, int ldda, std::complex<double> const* dtau,
    std::complex<double> const* dC, int lddc, int* lwork )
{
    return cusolverDnZunmqr_bufferSize(
        solver, side, trans, m, n, k,
        (cuDoubleComplex const*) dA, ldda,
        (cuDoubleComplex const*) dtau,
        (cuDoubleComplex const*) dC, lddc, lwork );
}

//------------------------------------------------------------------------------
// Intermediate wrappers around cuSolver to deal with precisions.
// Real uses ormqr, complex uses unmqr.
cusolverStatus_t cusolver_unmqr(
    cusolverDnHandle_t solver, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k,
    float const* dA, int ldda, float const* dtau,
    float* dC, int lddc,
    float* dev_work, int lwork, int* info )
{
    return cusolverDnSormqr(
        solver, side, trans, m, n, k, dA, ldda, dtau, dC, lddc,
        dev_work, lwork, info );
}

//----------
cusolverStatus_t cusolver_unmqr(
    cusolverDnHandle_t solver, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k,
    double const* dA, int ldda, double const* dtau,
    double* dC, int lddc,
    double* dev_work, int lwork, int* info )
{
    return cusolverDnDormqr(
        solver, side, trans, m, n, k, dA, ldda, dtau, dC, lddc,
        dev_work, lwork, info );
}

//----------
cusolverStatus_t cusolver_unmqr(
    cusolverDnHandle_t solver, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k,
    std::complex<float> const* dA, int ldda, std::complex<float> const* dtau,
    std::complex<float>* dC, int lddc,
    std::complex<float>* dev_work, int lwork, int* info )
{
    return cusolverDnCunmqr(
        solver, side, trans, m, n, k,
        (cuFloatComplex const*) dA, ldda,
        (cuFloatComplex const*) dtau,
        (cuFloatComplex*) dC, lddc,
        (cuFloatComplex*) dev_work, lwork, info );
}

//----------
cusolverStatus_t cusolver_unmqr(
    cusolverDnHandle_t solver, cublasSideMode_t side, cublasOperation_t trans,
    int m, int n, int k,
    std::complex<double> const* dA, int ldda, std::complex<double> const* dtau,
    std::complex<double>* dC, int lddc,
    std::complex<double>* dev_work, int lwork, int* info )
{
    return cusolverDnZunmqr(
        solver, side, trans, m, n, k,
        (cuDoubleComplex const*) dA, ldda,
        (cuDoubleComplex const*) dtau,
        (cuDoubleComplex*) dC, lddc,
        (cuDoubleComplex*) dev_work, lwork, info );
}

//------------------------------------------------------------------------------
// Wrapper around cuSolver workspace query.
template <typename scalar_t>
void unmqr_work_size_bytes(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k,
    scalar_t const* dA, int64_t ldda, scalar_t const* dtau,
    scalar_t* dC, int64_t lddc,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    // for real, map ConjTrans to Trans; complex doesn't take Trans.
    if (! blas::is_complex_v<scalar_t> && trans == Op::ConjTrans)
        trans = Op::Trans;
    lapack_error_if( blas::is_complex_v<scalar_t> && trans == Op::Trans );

    int m_    = to_cusolver_int( m );
    int n_    = to_cusolver_int( n );
    int k_    = to_cusolver_int( k );
    int ldda_ = to_cusolver_int( ldda );
    int lddc_ = to_cusolver_int( lddc );

    auto solver = queue.solver();

    // for cuda, rocm, call set_device; for oneapi, do nothing.
    blas::internal_set_device( queue.device() );

    // query for workspace size
    int lwork;
    blas_dev_call(
        cusolver_unmqr_bufferSize(
            solver,
            blas::internal::side2cublas( side ),
            blas::internal::op2cublas( trans ),
            m_, n_, k_, dA, ldda_, dtau, dC, lddc_, &lwork ));
    *dev_work_size = lwork * sizeof(scalar_t);
    *host_work_size = 0;
}

//------------------------------------------------------------------------------
// Wrapper around cuSolver.
// This is async. Once finished, the return info is in dev_info on the device.
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

    int m_    = to_cusolver_int( m );
    int n_    = to_cusolver_int( n );
    int k_    = to_cusolver_int( k );
    int ldda_ = to_cusolver_int( ldda );
    int lddc_ = to_cusolver_int( lddc );

    // A workspace larger than cuSolver can address is fine; use what fits.
    int lwork = int( blas::min( dev_work_size / sizeof(scalar_t),
                                size_t( std::numeric_limits<int>::max() ) ) );

    auto solver = queue.solver();

    // for cuda, rocm, call set_device; for oneapi, do nothing.
    blas::internal_set_device( queue.device() );

    // launch kernel
    blas_dev_call(
        cusolver_unmqr(
            solver,
            blas::internal::side2cublas( side ),
            blas::internal::op2cublas( trans ),
            m_, n_, k_, dA, ldda_, dtau, dC, lddc_,
            (scalar_t*) dev_work, lwork, dev_info ));
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

#endif // LAPACK_HAVE_CUBLAS
