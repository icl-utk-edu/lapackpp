// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_DEVICE_HH
#define LAPACK_DEVICE_HH

#include "blas/device.hh"
#include "lapack/util.hh"

#if defined(LAPACK_HAVE_CUBLAS)
    #include <cusolverDn.h>
#endif

namespace lapack {

// Since we pass pointers to these integers, their types have to match
// the vendor libraries.
#if defined(LAPACK_HAVE_CUBLAS)
    typedef int     device_info_int;
    typedef int64_t device_pivot_int;

#elif defined(LAPACK_HAVE_ROCBLAS)
    typedef rocblas_int device_info_int;
    typedef rocblas_int device_pivot_int;

#else
    typedef int64_t device_info_int;   ///< int type for returned info
    typedef int64_t device_pivot_int;  ///< int type for pivot vector (getrf, etc.)
#endif

//------------------------------------------------------------------------------
class Queue: public blas::Queue
{
public:
    Queue( int device=-1, int64_t batch_size=30000 )
      : blas::Queue( device, batch_size )
        #if defined(LAPACK_HAVE_CUBLAS)
            , solver_( nullptr )
            , solver_params_( nullptr )
        #endif
    {}

    ~Queue()
    {
        #if defined(LAPACK_HAVE_CUBLAS)
            if (solver_params_) {
                cusolverDnDestroyParams( solver_params_ );
                solver_params_ = nullptr;
            }
            if (solver_) {
                cusolverDnDestroy( solver_ );
                solver_ = nullptr;
            }
        #endif
    }

    // Disable copying; must construct anew.
    Queue( Queue const& ) = delete;
    Queue& operator=( Queue const& ) = delete;

    #if defined(LAPACK_HAVE_CUBLAS)
        /// @return cuSolver handle, allocating it on first use.
        cusolverDnHandle_t solver()
        {
            if (solver_ == nullptr) {
                // todo: error handler
                cusolverStatus_t status;
                status = cusolverDnCreate( &solver_ );
                assert( status == CUSOLVER_STATUS_SUCCESS );

                assert( stream() != nullptr );
                status = cusolverDnSetStream( solver_, stream() );
                assert( status == CUSOLVER_STATUS_SUCCESS );
            }
            return solver_;
        }

        /// @return cuSolver params, allocating it on first use.
        cusolverDnParams_t solver_params()
        {
            if (solver_params_ == nullptr) {
                // todo: error handler
                cusolverStatus_t status;
                status = cusolverDnCreateParams( &solver_params_ );
                assert( status == CUSOLVER_STATUS_SUCCESS );
            }
            return solver_params_;
        }
    #endif

private:
    #if defined(LAPACK_HAVE_CUBLAS)
        cusolverDnHandle_t solver_;
        cusolverDnParams_t solver_params_;
    #endif
};

//------------------------------------------------------------------------------
template <typename scalar_t>
void potrf(
    lapack::Uplo uplo, int64_t n,
    scalar_t* dA, int64_t ldda,
    device_info_int* dev_info, lapack::Queue& queue );

//------------------------------------------------------------------------------
template <typename scalar_t>
void getrf_work_size_bytes(
    int64_t m, int64_t n,
    scalar_t* dA, int64_t ldda,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template <typename scalar_t>
void getrf(
    int64_t m, int64_t n,
    scalar_t* dA, int64_t ldda, device_pivot_int* dev_ipiv,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

//------------------------------------------------------------------------------
template <typename scalar_t>
void geqrf_work_size_bytes(
    int64_t m, int64_t n,
    scalar_t* dA, int64_t ldda,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template <typename scalar_t>
void geqrf(
    int64_t m, int64_t n,
    scalar_t* dA, int64_t ldda, scalar_t* dtau,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

}  // namespace lapack

#endif // LAPACK_DEVICE_HH
