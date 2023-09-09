// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack/defines.h"

#if defined(LAPACK_HAVE_CUBLAS)

#include "lapack/device.hh"
#include "cuda_common.hh"

//==============================================================================
namespace blas {
namespace internal {

cublasFillMode_t uplo2cublas(blas::Uplo uplo);

} // namespace internal
} // namespace blas

//==============================================================================
namespace lapack {

//------------------------------------------------------------------------------
// Intermediate wrappers around cuSolver to deal with precisions.
cusolverStatus_t cusolver_heevd_bufferSize(
    cusolverDnHandle_t solver, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, float* dA,
    int ldda, float* dW, int* lwork )
{
    return cusolverDnSsyevd_bufferSize(
        solver, jobz, uplo, n, dA, ldda, dW, lwork );
}

//----------
cusolverStatus_t cusolver_heevd_bufferSize(
    cusolverDnHandle_t solver, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, double* dA,
    int ldda, double* dW, int* lwork )
{
    return cusolverDnDsyevd_bufferSize(
        solver, jobz, uplo, n, dA, ldda, dW, lwork );
}

//----------
cusolverStatus_t cusolver_heevd_bufferSize(
    cusolverDnHandle_t solver, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, std::complex<float>* dA,
    int ldda, float* dW, int* lwork )
{
    return cusolverDnCheevd_bufferSize(
        solver, jobz, uplo, n,
        (cuFloatComplex*) dA, ldda, dW, lwork );
}

//----------
cusolverStatus_t cusolver_heevd_bufferSize(
    cusolverDnHandle_t solver, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, std::complex<double>* dA,
    int ldda, double* dW, int* lwork )
{
    return cusolverDnZheevd_bufferSize(
        solver, jobz, uplo, n,
        (cuDoubleComplex*) dA, ldda, dW, lwork );
}

//------------------------------------------------------------------------------
// Intermediate wrappers around cuSolver to deal with precisions.
cusolverStatus_t cusolver_heevd(
    cusolverDnHandle_t solver, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, float* dA,
    int ldda, float* dW, float* dev_work, int lwork, int* info )
{
    return cusolverDnSsyevd(
        solver, jobz, uplo, n, dA, ldda, dW, dev_work, lwork, info );
}

//----------
cusolverStatus_t cusolver_heevd(
    cusolverDnHandle_t solver, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, double* dA,
    int ldda, double* dW, double* dev_work, int lwork, int* info )
{
    return cusolverDnDsyevd(
        solver, jobz, uplo, n, dA, ldda, dW, dev_work, lwork, info );
}

//----------
cusolverStatus_t cusolver_heevd(
    cusolverDnHandle_t solver, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, std::complex<float>* dA,
    int ldda, float* dW, float* dev_work, int lwork, int* info )
{
    return cusolverDnCheevd(
        solver, jobz, uplo, n,
        (cuFloatComplex*) dA, ldda,
        dW,
        (cuFloatComplex*) dev_work, lwork, info );
}

//----------
cusolverStatus_t cusolver_heevd(
    cusolverDnHandle_t solver, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int n, std::complex<double>* dA,
    int ldda, double* dW, double* dev_work, int lwork, int* info )
{
    return cusolverDnZheevd(
        solver, jobz, uplo, n,
        (cuDoubleComplex*) dA, ldda,
        dW,
        (cuDoubleComplex*) dev_work, lwork, info );
}

//------------------------------------------------------------------------------
// Wrapper around cuSolver workspace query.
// dA is only for templating scalar_t; it isn't referenced.
template <typename scalar_t>
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo,
    int64_t n, scalar_t* dA, int64_t ldda, blas::real_type<scalar_t>* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    using real_t = blas::real_type<scalar_t>;
    auto solver = queue.solver();

    // for cuda, rocm, call set_device; for oneapi, do nothing.
    blas::internal_set_device( queue.device() );

    // query for workspace size
    #if CUSOLVER_VERSION >= 11000
        auto params = queue.solver_params();
        blas_dev_call(
            cusolverDnXsyevd_bufferSize(
                solver, params, job2eigmode_cusolver(jobz),
                blas::internal::uplo2cublas(uplo), n,
                CudaTraits<scalar_t>::datatype, dA, ldda,
                CudaTraits<real_t>  ::datatype, dW,
                CudaTraits<scalar_t>::datatype,
                dev_work_size, host_work_size));
    #else
        int lwork;
        blas_dev_call(
            cusolver_heevd_bufferSize( solver, job2eigmode_cusolver(jobz), blas::internal::uplo2cublas(uplo), n, dA, ldda, dW, &lwork ));
        *dev_work_size = lwork * sizeof(scalar_t);
        *host_work_size = 0;
    #endif
}

//------------------------------------------------------------------------------
// Wrapper around cuSolver.
// This is async. Once finished, the return info is in dev_info on the device.
template <typename scalar_t>
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    scalar_t* dA, int64_t ldda, blas::real_type<scalar_t>* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue )
{
    using real_t = blas::real_type<scalar_t>;
    auto solver = queue.solver();

    // for cuda, rocm, call set_device; for oneapi, do nothing.
    blas::internal_set_device( queue.device() );

    // launch kernel
    #if CUSOLVER_VERSION >= 11000
        auto params = queue.solver_params();
        blas_dev_call(
            cusolverDnXsyevd(
                solver, params, job2eigmode_cusolver(jobz),
                blas::internal::uplo2cublas(uplo), n,
                CudaTraits<scalar_t>::datatype, dA, ldda,
                CudaTraits<real_t>  ::datatype, dW,
                CudaTraits<scalar_t>::datatype,
                dev_work, dev_work_size,
                host_work, host_work_size, dev_info ));
    #else
        int lwork = dev_work_size / sizeof(scalar_t);
        blas_dev_call(
            cusolver_heevd(
                solver, params, job2eigmode_cusolver(jobz), blas::internal::uplo2cublas(uplo), n, dA, ldda, dW,
                (scalar_t*) dev_work, lwork, dev_info ));
    #endif
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo,
    int64_t n, float* dA, int64_t ldda, float* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo,
    int64_t n, double* dA, int64_t ldda, double* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo,
    int64_t n, std::complex<float>* dA, int64_t ldda, float* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo,
    int64_t n, std::complex<double>* dA, int64_t ldda, double* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

//--------------------
template
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    float* dA, int64_t ldda, float* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

template
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    double* dA, int64_t ldda, double* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

template
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    std::complex<float>* dA, int64_t ldda, float* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

template
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    std::complex<double>* dA, int64_t ldda, double* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

} // namespace lapack

#endif // LAPACK_HAVE_CUBLAS

