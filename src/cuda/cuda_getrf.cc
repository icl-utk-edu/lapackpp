// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack/defines.h"

#if defined(LAPACK_HAVE_CUBLAS)

#include "lapack/device.hh"
#include "cuda_common.hh"

//==============================================================================
namespace lapack {

//------------------------------------------------------------------------------
// Intermediate wrappers around cuSolver to deal with precisions.
cusolverStatus_t cusolver_getrf_bufferSize(
    cusolverDnHandle_t solver, int m, int n,
    float* dA, int ldda, int* lwork )
{
    return cusolverDnSgetrf_bufferSize(
        solver, m, n, dA, ldda, lwork );
}

//----------
cusolverStatus_t cusolver_getrf_bufferSize(
    cusolverDnHandle_t solver, int m, int n,
    double* dA, int ldda, int* lwork )
{
    return cusolverDnDgetrf_bufferSize(
        solver, m, n, dA, ldda, lwork );
}

//----------
cusolverStatus_t cusolver_getrf_bufferSize(
    cusolverDnHandle_t solver, int m, int n,
    std::complex<float>* dA, int ldda, int* lwork )
{
    return cusolverDnCgetrf_bufferSize(
        solver, m, n,
        (cuFloatComplex*) dA, ldda, lwork );
}

//----------
cusolverStatus_t cusolver_getrf_bufferSize(
    cusolverDnHandle_t solver, int m, int n,
    std::complex<double>* dA, int ldda, int* lwork )
{
    return cusolverDnZgetrf_bufferSize(
        solver, m, n,
        (cuDoubleComplex*) dA, ldda, lwork );
}

//------------------------------------------------------------------------------
// Intermediate wrappers around cuSolver to deal with precisions.
cusolverStatus_t cusolver_getrf(
    cusolverDnHandle_t solver, int m, int n,
    float* dA, int ldda, int* dipiv,
    float* dev_work, int* info )
{
    return cusolverDnSgetrf(
        solver, m, n, dA, ldda, dev_work, dipiv, info );
}

//----------
cusolverStatus_t cusolver_getrf(
    cusolverDnHandle_t solver, int m, int n,
    double* dA, int ldda, int* dipiv,
    double* dev_work, int* info )
{
    return cusolverDnDgetrf(
        solver, m, n, dA, ldda, dev_work, dipiv, info );
}

//----------
cusolverStatus_t cusolver_getrf(
    cusolverDnHandle_t solver, int m, int n,
    std::complex<float>* dA, int ldda, int* dipiv,
    std::complex<float>* dev_work, int* info )
{
    return cusolverDnCgetrf(
        solver, m, n,
        (cuFloatComplex*) dA, ldda,
        (cuFloatComplex*) dev_work, dipiv, info );
}

//----------
cusolverStatus_t cusolver_getrf(
    cusolverDnHandle_t solver, int m, int n,
    std::complex<double>* dA, int ldda, int* dipiv,
    std::complex<double>* dev_work, int* info )
{
    return cusolverDnZgetrf(
        solver, m, n,
        (cuDoubleComplex*) dA, ldda,
        (cuDoubleComplex*) dev_work, dipiv, info );
}

//------------------------------------------------------------------------------
// Wrapper around cuSolver workspace query.
// dA is only for templating scalar_t; it isn't referenced.
template <typename scalar_t>
void getrf_work_size_bytes(
    int64_t m, int64_t n,
    scalar_t* dA, int64_t ldda,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    auto solver = queue.solver();

    // for cuda, rocm, call set_device; for oneapi, do nothing.
    blas::internal_set_device( queue.device() );

    // query for workspace size
    #if CUSOLVER_VERSION >= 11000
        auto params = queue.solver_params();
        blas_dev_call(
            cusolverDnXgetrf_bufferSize(
                solver, params, m, n,
                CudaTraits<scalar_t>::datatype, dA, ldda,
                CudaTraits<scalar_t>::datatype, dev_work_size, host_work_size ));
    #else
        int lwork;
        blas_dev_call(
            cusolver_getrf_bufferSize( solver, m, n, dA, ldda, &lwork ));
        *dev_work_size = lwork * sizeof(scalar_t);
        *host_work_size = 0;
    #endif

    //printf( "%s m %lld, n %lld, ldda %lld, buffer ldda*n %lld, dev %lld, host %lld\n",
    //        __func__, llong( m ), llong( n ), llong( ldda ), llong( ldda * n ),
    //        llong(  *dev_work_size/sizeof(scalar_t) ),
    //        llong( *host_work_size/sizeof(scalar_t) ) );
}

//------------------------------------------------------------------------------
// Wrapper around cuSolver.
// This is async. Once finished, the return info is in dev_info on the device.
template <typename scalar_t>
void getrf(
    int64_t m, int64_t n,
    scalar_t* dA, int64_t ldda, device_pivot_int* dipiv,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue )
{
    auto solver = queue.solver();

    // for cuda, rocm, call set_device; for oneapi, do nothing.
    blas::internal_set_device( queue.device() );

    // launch kernel
    #if CUSOLVER_VERSION >= 11000
        auto params = queue.solver_params();
        blas_dev_call(
            cusolverDnXgetrf(
                solver, params, m, n,
                CudaTraits<scalar_t>::datatype, dA, ldda, dipiv,
                CudaTraits<scalar_t>::datatype,
                dev_work, dev_work_size,
                host_work, host_work_size, dev_info ));
    #else
        blas_dev_call(
            cusolver_getrf(
                solver, m, n, dA, ldda, dipiv,
                (scalar_t*) dev_work, dev_info ));
    #endif
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void getrf_work_size_bytes(
    int64_t m, int64_t n,
    float* dA, int64_t ldda,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template
void getrf_work_size_bytes(
    int64_t m, int64_t n,
    double* dA, int64_t ldda,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template
void getrf_work_size_bytes(
    int64_t m, int64_t n,
    std::complex<float>* dA, int64_t ldda,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template
void getrf_work_size_bytes(
    int64_t m, int64_t n,
    std::complex<double>* dA, int64_t ldda,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

//--------------------
template
void getrf(
    int64_t m, int64_t n,
    float* dA, int64_t ldda, device_pivot_int* dipiv,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

template
void getrf(
    int64_t m, int64_t n,
    double* dA, int64_t ldda, device_pivot_int* dipiv,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

template
void getrf(
    int64_t m, int64_t n,
    std::complex<float>* dA, int64_t ldda, device_pivot_int* dipiv,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

template
void getrf(
    int64_t m, int64_t n,
    std::complex<double>* dA, int64_t ldda, device_pivot_int* dipiv,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

} // namespace lapack

#endif // LAPACK_HAVE_CUBLAS
