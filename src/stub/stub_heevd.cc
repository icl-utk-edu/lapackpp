// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack/defines.h"

#if ! (defined(LAPACK_HAVE_ROCBLAS) || defined(LAPACK_HAVE_CUBLAS) || defined(LAPACK_HAVE_SYCL))

#include "lapack/device.hh"


//==============================================================================
namespace lapack {

//------------------------------------------------------------------------------
// Wrapper around workspace query.
// dA is only for templating scalar_t; it isn't referenced.
template <typename scalar_t>
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo,
    int64_t n, scalar_t* dA, int64_t ldda, blas::real_type<scalar_t>* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    *dev_work_size  = 0;
    *host_work_size = 0;
}

//------------------------------------------------------------------------------
// Wrapper stub.
// This is async. Once finished, the return info is in dev_info on the device.
template <typename scalar_t>
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    scalar_t* dA, int64_t ldda, blas::real_type<scalar_t>* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue )
{
    throw lapack::Error( "device LAPACK not available", __func__ );
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

