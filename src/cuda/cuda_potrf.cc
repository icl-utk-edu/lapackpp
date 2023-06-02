// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack/defines.h"

#if defined(LAPACK_HAVE_CUBLAS)

#include "lapack/device.hh"
#include "cuda_common.hh"

//==============================================================================
// todo: put into BLAS++ header somewhere.
// changed from blas::device to blas::internal in 5ca8ad35 2022-11-28

namespace blas {
namespace internal {

cublasFillMode_t uplo2cublas( blas::Uplo uplo );

} // namespace internal
} // namespace blas

//==============================================================================
namespace lapack {

//------------------------------------------------------------------------------
// Intermediate wrappers around cuSolver to deal with precisions.
cusolverStatus_t cusolver_potrf_bufferSize(
    cusolverDnHandle_t solver, cublasFillMode_t uplo, int n,
    float* dA, int ldda, int* lwork )
{
    return cusolverDnSpotrf_bufferSize(
        solver, uplo, n, dA, ldda, lwork );
}

//----------
cusolverStatus_t cusolver_potrf_bufferSize(
    cusolverDnHandle_t solver, cublasFillMode_t uplo, int n,
    double* dA, int ldda, int* lwork )
{
    return cusolverDnDpotrf_bufferSize(
        solver, uplo, n, dA, ldda, lwork );
}

//----------
cusolverStatus_t cusolver_potrf_bufferSize(
    cusolverDnHandle_t solver, cublasFillMode_t uplo, int n,
    std::complex<float>* dA, int ldda, int* lwork )
{
    return cusolverDnCpotrf_bufferSize(
        solver, uplo, n,
        (cuFloatComplex*) dA, ldda, lwork );
}

//----------
cusolverStatus_t cusolver_potrf_bufferSize(
    cusolverDnHandle_t solver, cublasFillMode_t uplo, int n,
    std::complex<double>* dA, int ldda, int* lwork )
{
    return cusolverDnZpotrf_bufferSize(
        solver, uplo, n,
        (cuDoubleComplex*) dA, ldda, lwork );
}

//------------------------------------------------------------------------------
// Intermediate wrappers around cuSolver to deal with precisions.
cusolverStatus_t cusolver_potrf(
    cusolverDnHandle_t solver, cublasFillMode_t uplo, int n,
    float* dA, int ldda,
    float* dev_work, int lwork, int* info )
{
    return cusolverDnSpotrf(
        solver, uplo, n, dA, ldda, dev_work, lwork, info );
}

//----------
cusolverStatus_t cusolver_potrf(
    cusolverDnHandle_t solver, cublasFillMode_t uplo, int n,
    double* dA, int ldda,
    double* dev_work, int lwork, int* info )
{
    return cusolverDnDpotrf(
        solver, uplo, n, dA, ldda, dev_work, lwork, info );
}

//----------
cusolverStatus_t cusolver_potrf(
    cusolverDnHandle_t solver, cublasFillMode_t uplo, int n,
    std::complex<float>* dA, int ldda,
    std::complex<float>* dev_work, int lwork, int* info )
{
    return cusolverDnCpotrf(
        solver, uplo, n,
        (cuFloatComplex*) dA, ldda,
        (cuFloatComplex*) dev_work, lwork, info );
}

//----------
cusolverStatus_t cusolver_potrf(
    cusolverDnHandle_t solver, cublasFillMode_t uplo, int n,
    std::complex<double>* dA, int ldda,
    std::complex<double>* dev_work, int lwork, int* info )
{
    return cusolverDnZpotrf(
        solver, uplo, n,
        (cuDoubleComplex*) dA, ldda,
        (cuDoubleComplex*) dev_work, lwork, info );
}

//------------------------------------------------------------------------------
// Wrapper around cuSolver workspace query.
template <typename scalar_t>
void potrf_work_size_bytes(
    lapack::Uplo uplo, int64_t n,
    scalar_t* dA, int64_t ldda,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    auto solver = queue.solver();
    auto uplo_ = blas::internal::uplo2cublas( uplo );

    // query for workspace size
    #if CUSOLVER_VERSION >= 11000
        auto params = queue.solver_params();
        blas_dev_call(
            cusolverDnXpotrf_bufferSize(
                solver, params, uplo_, n,
                CudaTraits<scalar_t>::datatype, dA, ldda,
                CudaTraits<scalar_t>::datatype, dev_work_size, host_work_size ));
    #else
        int lwork;
        blas_dev_call(
            cusolver_potrf_bufferSize( solver, uplo_, n, dA, ldda, &lwork ));
        *dev_work_size = lwork * sizeof(scalar_t);
        *host_work_size = 0;
    #endif
}

//------------------------------------------------------------------------------
// Wrapper around cuSolver.
// This is async. Once finished, the return info is in dev_info on the device.
template <typename scalar_t>
void potrf(
    lapack::Uplo uplo, int64_t n,
    scalar_t* dA, int64_t ldda,
    device_info_int* dev_info, lapack::Queue& queue )
{
    // todo: check for overflow
    auto solver = queue.solver();
    auto uplo_ = blas::internal::uplo2cublas( uplo );

    // for cuda, rocm, call set_device; for oneapi, do nothing.
    blas::internal_set_device( queue.device() );

    // query for workspace size
    size_t dev_work_size, host_work_size;
    potrf_work_size_bytes(
        uplo, n, dA, ldda, &dev_work_size, &host_work_size, queue );

    // alloc workspace in queue
    queue.work_ensure_size< char >( dev_work_size );  // syncs if needed
    void* dev_work = queue.work();
    blas_error_if( host_work_size != 0 );

    // launch kernel
    #if CUSOLVER_VERSION >= 11000
        auto params = queue.solver_params();
        blas_dev_call(
            cusolverDnXpotrf(
                solver, params, uplo_, n,
                CudaTraits<scalar_t>::datatype, dA, ldda,
                CudaTraits<scalar_t>::datatype,
                dev_work, dev_work_size,
                nullptr, 0,  // host work assumed 0
                dev_info ));
    #else
        int lwork = dev_work_size / sizeof(scalar_t);
        blas_dev_call(
            cusolver_potrf(
                solver, uplo_, n, dA, ldda,
                (scalar_t*) dev_work, lwork, dev_info ));
    #endif
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void potrf(
    lapack::Uplo uplo, int64_t n,
    float* dA, int64_t ldda,
    device_info_int* dev_info, lapack::Queue& queue );

template
void potrf(
    lapack::Uplo uplo, int64_t n,
    double* dA, int64_t ldda,
    device_info_int* dev_info, lapack::Queue& queue );

template
void potrf(
    lapack::Uplo uplo, int64_t n,
    std::complex<float>* dA, int64_t ldda,
    device_info_int* dev_info, lapack::Queue& queue );

template
void potrf(
    lapack::Uplo uplo, int64_t n,
    std::complex<double>* dA, int64_t ldda,
    device_info_int* dev_info, lapack::Queue& queue );

} // namespace lapack

#endif // LAPACK_HAVE_CUBLAS
