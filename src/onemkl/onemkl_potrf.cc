// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack/defines.h"

#if defined(LAPACK_HAVE_SYCL)

#include "onemkl_common.hh"

//==============================================================================
// todo: put into BLAS++ header somewhere.

namespace blas {
namespace internal {

oneapi::mkl::uplo uplo2onemkl(blas::Uplo uplo);

} // namespace internal
} // namespace blas

//==============================================================================
namespace lapack {

//------------------------------------------------------------------------------
// Wrapper around workspace query.
template <typename scalar_t>
void potrf_work_size_bytes(
    lapack::Uplo uplo, int64_t n,
    scalar_t* dA, int64_t ldda,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    auto solver = queue.stream();
    auto uplo_ = blas::internal::uplo2onemkl( uplo );

    // for cuda, rocm, call set_device; for oneapi, do nothing.
    blas::internal_set_device( queue.device() );

    // query for workspace size
    int64_t lwork = 0;
    blas_dev_call(
        lwork = oneapi::mkl::lapack::potrf_scratchpad_size<scalar_t>(
            solver, uplo_, n, ldda ));
    *dev_work_size = lwork * sizeof(scalar_t);
    *host_work_size = 0;
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
    auto solver = queue.stream();
    auto uplo_ = blas::internal::uplo2onemkl( uplo );

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
    int64_t lwork = dev_work_size/sizeof(scalar_t);
    blas_dev_call(
        oneapi::mkl::lapack::potrf(
            solver, uplo_, n, dA, ldda, (scalar_t*)dev_work, lwork ));

    // todo: default info returned
    blas::device_memset( dev_info, 0, 1, queue );
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

#endif // LAPACK_HAVE_SYCL
