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
// Wrapper stub.
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
    throw lapack::Error( "device LAPACK not available", __func__ );
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

#endif // ! (LAPACK_HAVE_ROCBLAS || LAPACK_HAVE_CUBLAS || LAPACK_HAVE_SYCL)
