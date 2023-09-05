// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack/defines.h"

#if defined(LAPACK_HAVE_SYCL)

#include "lapack/device.hh"
#include "onemkl_common.hh"

//==============================================================================
namespace blas {
namespace internal {

// Access function blas::internal::uplo2onemkl()
oneapi::mkl::uplo uplo2onemkl(blas::Uplo uplo);

} // namespace internal
} // namespace blas

//==============================================================================
namespace lapack {

//==============================================================================
// Specified scalar_t float -> syevd
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    float* dA, int64_t ldda, float* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    // query for workspace size
    auto solver = queue.stream();
    int lwork = 0;
    lwork = oneapi::mkl::lapack::syevd_scratchpad_size<float>(
        solver, jobz2onemkl(jobz), blas::internal::uplo2onemkl(uplo), n, ldda );
    *dev_work_size = lwork * sizeof(float);
    *host_work_size = 0;
}

//------------------------------------------------------------------------------
// Specified scalar_t double -> syevd
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    double* dA, int64_t ldda, double* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    // query for workspace size
    auto solver = queue.stream();
    int lwork = 0;
    lwork = oneapi::mkl::lapack::syevd_scratchpad_size<double>(
        solver, jobz2onemkl(jobz), blas::internal::uplo2onemkl(uplo), n, ldda );
    *dev_work_size = lwork * sizeof(double);
    *host_work_size = 0;
}

//------------------------------------------------------------------------------
// Specified scalar_t std::complex<float> -> heevd
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    std::complex<float>* dA, int64_t ldda, float* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    // query for workspace size
    auto solver = queue.stream();
    int lwork = 0;
    lwork = oneapi::mkl::lapack::heevd_scratchpad_size<std::complex<float>>(
        solver, jobz2onemkl(jobz), blas::internal::uplo2onemkl(uplo), n, ldda );
    *dev_work_size = lwork * sizeof(std::complex<float>);
    *host_work_size = 0;
}

//------------------------------------------------------------------------------
// Specified scalar_t std::complex<double> -> heevd
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    std::complex<double>* dA, int64_t ldda, double* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    // query for workspace size
    auto solver = queue.stream();
    int lwork = 0;
    lwork = oneapi::mkl::lapack::heevd_scratchpad_size<std::complex<double>>(
        solver, jobz2onemkl(jobz), blas::internal::uplo2onemkl(uplo), n, ldda );
    *dev_work_size = lwork * sizeof(std::complex<double>);
    *host_work_size = 0;
}

//------------------------------------------------------------------------------
// Templated for scalar_t
template <typename scalar_t>
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    scalar_t* dA, int64_t ldda, blas::real_type<scalar_t>* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    // call scalar_t specified routines
    heevd_work_size_bytes(
        jobz, uplo, n, dA, ldda, dW, dev_work_size, host_work_size, queue );
}

//==============================================================================
// Specified scalar_t float -> syevd
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    float* dA, int64_t ldda, float* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue )
{
    // launch kernel
    auto solver = queue.stream();
    int lwork = dev_work_size / sizeof(float);
    blas_dev_call(
        oneapi::mkl::lapack::syevd(
            solver, jobz2onemkl(jobz), blas::internal::uplo2onemkl(uplo), n,
            dA, ldda, dW, (float*) dev_work, lwork ));

    // default info returned
    blas::device_memset( dev_info, 0, 1, queue );
}

//------------------------------------------------------------------------------
// Specified scalar_t double -> syevd
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    double* dA, int64_t ldda, double* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue )
{
    // launch kernel
    auto solver = queue.stream();
    int lwork = dev_work_size / sizeof(double);
    blas_dev_call(
        oneapi::mkl::lapack::syevd(
            solver, jobz2onemkl(jobz), blas::internal::uplo2onemkl(uplo), n,
            dA, ldda, dW, (double*) dev_work, lwork ));

    // default info returned
    blas::device_memset( dev_info, 0, 1, queue );
}

//------------------------------------------------------------------------------
// Specified scalar_t std::complex<float> -> heevd
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    std::complex<float>* dA, int64_t ldda, float* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue )
{
    // launch kernel
    auto solver = queue.stream();
    int lwork = dev_work_size / sizeof(std::complex<float>);
    blas_dev_call(
        oneapi::mkl::lapack::heevd(
            solver, jobz2onemkl(jobz), blas::internal::uplo2onemkl(uplo), n,
            dA, ldda, dW, (std::complex<float>*) dev_work, lwork ));

    // default info returned
    blas::device_memset( dev_info, 0, 1, queue );
}

//------------------------------------------------------------------------------
// Specified scalar_t std::complex<double> -> heevd
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    std::complex<double>* dA, int64_t ldda, double* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue )
{
    // launch kernel
    auto solver = queue.stream();
    int lwork = dev_work_size / sizeof(std::complex<double>);
    blas_dev_call(
        oneapi::mkl::lapack::heevd(
            solver, jobz2onemkl(jobz), blas::internal::uplo2onemkl(uplo), n,
            dA, ldda, dW, (std::complex<double>*) dev_work, lwork ));

    // default info returned
    blas::device_memset( dev_info, 0, 1, queue );
}

//------------------------------------------------------------------------------
// Templated for scalar_t.  This is an async call, the return info is
// in dev_info on the device.
// Currently dev_info is always set to default 0.
template <typename scalar_t>
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    scalar_t* dA, int64_t ldda, blas::real_type<scalar_t>* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue )
{
    // call scalar_t specified routines
    heevd(
        jobz, uplo, n, dA, ldda, dW, dev_work, dev_work_size,
        host_work, host_work_size, dev_info, queue );
}

//==============================================================================
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

#endif // LAPACK_HAVE_SYCL
