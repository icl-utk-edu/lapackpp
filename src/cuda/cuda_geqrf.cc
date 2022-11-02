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
cusolverStatus_t cusolver_geqrf_bufferSize(
    cusolverDnHandle_t solver, int m, int n,
    float* dA, int ldda, int* lwork )
{
    return cusolverDnSgeqrf_bufferSize(
        solver, m, n, dA, ldda, lwork );
}

//----------
cusolverStatus_t cusolver_geqrf_bufferSize(
    cusolverDnHandle_t solver, int m, int n,
    double* dA, int ldda, int* lwork )
{
    return cusolverDnDgeqrf_bufferSize(
        solver, m, n, dA, ldda, lwork );
}

//----------
cusolverStatus_t cusolver_geqrf_bufferSize(
    cusolverDnHandle_t solver, int m, int n,
    std::complex<float>* dA, int ldda, int* lwork )
{
    return cusolverDnCgeqrf_bufferSize(
        solver, m, n,
        (cuFloatComplex*) dA, ldda, lwork );
}

//----------
cusolverStatus_t cusolver_geqrf_bufferSize(
    cusolverDnHandle_t solver, int m, int n,
    std::complex<double>* dA, int ldda, int* lwork )
{
    return cusolverDnZgeqrf_bufferSize(
        solver, m, n,
        (cuDoubleComplex*) dA, ldda, lwork );
}

//------------------------------------------------------------------------------
// Intermediate wrappers around cuSolver to deal with precisions.
cusolverStatus_t cusolver_geqrf(
    cusolverDnHandle_t solver, int m, int n,
    float* dA, int ldda, float* dtau,
    float* dev_work, int lwork, int* info )
{
    return cusolverDnSgeqrf(
        solver, m, n, dA, ldda, dtau, dev_work, lwork, info );
}

//----------
cusolverStatus_t cusolver_geqrf(
    cusolverDnHandle_t solver, int m, int n,
    double* dA, int ldda, double* dtau,
    double* dev_work, int lwork, int* info )
{
    return cusolverDnDgeqrf(
        solver, m, n, dA, ldda, dtau, dev_work, lwork, info );
}

//----------
cusolverStatus_t cusolver_geqrf(
    cusolverDnHandle_t solver, int m, int n,
    std::complex<float>* dA, int ldda, std::complex<float>* dtau,
    std::complex<float>* dev_work, int lwork, int* info )
{
    return cusolverDnCgeqrf(
        solver, m, n,
        (cuFloatComplex*) dA, ldda,
        (cuFloatComplex*) dtau,
        (cuFloatComplex*) dev_work, lwork, info );
}

//----------
cusolverStatus_t cusolver_geqrf(
    cusolverDnHandle_t solver, int m, int n,
    std::complex<double>* dA, int ldda, std::complex<double>* dtau,
    std::complex<double>* dev_work, int lwork, int* info )
{
    return cusolverDnZgeqrf(
        solver, m, n,
        (cuDoubleComplex*) dA, ldda,
        (cuDoubleComplex*) dtau,
        (cuDoubleComplex*) dev_work, lwork, info );
}

//------------------------------------------------------------------------------
// Wrapper around cuSolver workspace query.
// dA is only for templating scalar_t; it isn't referenced.
template <typename scalar_t>
void geqrf_work_size_bytes(
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
            cusolverDnXgeqrf_bufferSize(
                solver, params, m, n,
                CudaTraits<scalar_t>::datatype, dA, ldda,
                CudaTraits<scalar_t>::datatype, nullptr,
                CudaTraits<scalar_t>::datatype, dev_work_size, host_work_size ));
    #else
        int lwork;
        blas_dev_call(
            cusolver_geqrf_bufferSize( solver, m, n, dA, ldda, &lwork ));
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
void geqrf(
    int64_t m, int64_t n,
    scalar_t* dA, int64_t ldda, scalar_t* dtau,
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
            cusolverDnXgeqrf(
                solver, params, m, n,
                CudaTraits<scalar_t>::datatype, dA, ldda,
                CudaTraits<scalar_t>::datatype, dtau,
                CudaTraits<scalar_t>::datatype,
                dev_work, dev_work_size,
                host_work, host_work_size, dev_info ));
    #else
        int lwork = dev_work_size / sizeof(scalar_t);
        blas_dev_call(
            cusolver_geqrf(
                solver, m, n, dA, ldda, dtau,
                (scalar_t*) dev_work, lwork, dev_info ));
    #endif
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geqrf_work_size_bytes(
    int64_t m, int64_t n,
    float* dA, int64_t ldda,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template
void geqrf_work_size_bytes(
    int64_t m, int64_t n,
    double* dA, int64_t ldda,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template
void geqrf_work_size_bytes(
    int64_t m, int64_t n,
    std::complex<float>* dA, int64_t ldda,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template
void geqrf_work_size_bytes(
    int64_t m, int64_t n,
    std::complex<double>* dA, int64_t ldda,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

//--------------------
template
void geqrf(
    int64_t m, int64_t n,
    float* dA, int64_t ldda, float* dtau,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

template
void geqrf(
    int64_t m, int64_t n,
    double* dA, int64_t ldda, double* dtau,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

template
void geqrf(
    int64_t m, int64_t n,
    std::complex<float>* dA, int64_t ldda, std::complex<float>* dtau,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

template
void geqrf(
    int64_t m, int64_t n,
    std::complex<double>* dA, int64_t ldda, std::complex<double>* dtau,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

} // namespace lapack

#endif // LAPACK_HAVE_CUBLAS
