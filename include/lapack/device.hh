// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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
    typedef int device_info_int;
    #if CUSOLVER_VERSION >= 11000
        typedef int64_t device_pivot_int;
    #else
        typedef int device_pivot_int;
    #endif

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
    Queue()
      : blas::Queue()
        #if defined(LAPACK_HAVE_CUBLAS)
            , solver_( nullptr )
            #if CUSOLVER_VERSION >= 11000
                , solver_params_( nullptr )
            #endif
        #endif
    {}

    Queue( int device )
      : blas::Queue( device )
        #if defined(LAPACK_HAVE_CUBLAS)
            , solver_( nullptr )
            #if CUSOLVER_VERSION >= 11000
                , solver_params_( nullptr )
            #endif
        #endif
    {}

    ~Queue()
    {
        #if defined(LAPACK_HAVE_CUBLAS)
            blas::internal_set_device( device() );
            #if CUSOLVER_VERSION >= 11000
                if (solver_params_) {
                    cusolverDnDestroyParams( solver_params_ );
                    solver_params_ = nullptr;
                }
            #endif

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
                blas::internal_set_device( device() );
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

        #if CUSOLVER_VERSION >= 11000
            /// @return cuSolver params, allocating it on first use.
            cusolverDnParams_t solver_params()
            {
                if (solver_params_ == nullptr) {
                    blas::internal_set_device( device() );
                    // todo: error handler
                    cusolverStatus_t status;
                    status = cusolverDnCreateParams( &solver_params_ );
                    assert( status == CUSOLVER_STATUS_SUCCESS );
                }
                return solver_params_;
            }
        #endif
    #endif

private:
    #if defined(LAPACK_HAVE_CUBLAS)
        cusolverDnHandle_t solver_;
        #if CUSOLVER_VERSION >= 11000
            cusolverDnParams_t solver_params_;
        #endif
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

//------------------------------------------------------------------------------
template <typename scalar_t>
void heevd_work_size_bytes(
    lapack::Job jobz, lapack::Uplo uplo,
    int64_t n, scalar_t* dA, int64_t ldda, blas::real_type<scalar_t>* dW,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue );

template <typename scalar_t>
void heevd(
    lapack::Job jobz, lapack::Uplo uplo, int64_t n,
    scalar_t* dA, int64_t ldda, blas::real_type<scalar_t>* dW,
    void*  dev_work, size_t  dev_work_size,
    void* host_work, size_t host_work_size,
    device_info_int* dev_info, lapack::Queue& queue );

template <typename scalar_t>
void larfg(
    int64_t n,
    scalar_t* alpha,
    scalar_t* dx, int64_t incdx,
    scalar_t* tau,
    lapack::Queue& queue );

template <typename scalar_t>
int64_t tpqrt(
    int64_t m, int64_t n, int64_t l, int64_t nb,
    scalar_t* dA, int64_t ldda,
    scalar_t* dB, int64_t lddb,
    scalar_t* dT, int64_t lddt,
    lapack::Queue& queue );

template <typename scalar_t>
int64_t tpqrt2(
    int64_t m, int64_t n, int64_t l,
    scalar_t* dA, int64_t ldda,
    scalar_t* dB, int64_t lddb,
    scalar_t* dT, int64_t lddt,
    lapack::Queue& queue );

template <typename scalar_t>
void tprfb(
    lapack::Side side, lapack::Op trans,
    lapack::Direction direction, lapack::StoreV storev,
    int64_t m, int64_t n, int64_t k, int64_t l,
    scalar_t const* dV, int64_t ldv,
    scalar_t const* dT, int64_t ldt,
    scalar_t* dA, int64_t lda,
    scalar_t* dB, int64_t ldb,
    lapack::Queue& queue );

}  // namespace lapack

#endif // LAPACK_DEVICE_HH
