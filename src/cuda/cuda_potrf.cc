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

namespace blas {
namespace device {

cublasFillMode_t uplo2cublas( blas::Uplo uplo );

} // namespace device
} // namespace blas

//==============================================================================
namespace lapack {

//------------------------------------------------------------------------------
// Wrapper around cuSolver.
// This is async. Once finished, the return info is in dev_info on the device.
template <typename scalar_t>
void potrf(
    lapack::Uplo uplo, int64_t n,
    scalar_t* dA, int64_t ldda,
    device_info_int* dev_info, lapack::Queue& queue )
{
    auto solver = queue.solver();
    auto params = queue.solver_params();
    auto uplo_ = blas::device::uplo2cublas( uplo );

    // query for workspace size
    size_t dev_work_size, host_work_size;
    blas_dev_call(
        cusolverDnXpotrf_bufferSize(
            solver, params, uplo_, n,
            CudaTraits<scalar_t>::datatype, dA, ldda,
            CudaTraits<scalar_t>::datatype, &dev_work_size, &host_work_size ));

    //printf( "%s n %lld, ldda %lld, buffer ldda*n %lld, dev %lld, host %lld\n",
    //        __func__, llong( n ), llong( ldda ), llong( ldda * n ),
    //        llong(  dev_work_size/sizeof(scalar_t) ),
    //        llong( host_work_size/sizeof(scalar_t) ) );

    // alloc workspaces
    queue.work_resize< char >( dev_work_size );  // syncs if needed
    void* dwork = queue.work();
    blas_error_if( host_work_size != 0 );

    // launch kernel
    blas_dev_call(
        cusolverDnXpotrf(
            solver, params, uplo_, n,
            CudaTraits<scalar_t>::datatype, dA, ldda,
            CudaTraits<scalar_t>::datatype, dwork, dev_work_size, nullptr, 0,
            dev_info ));
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
