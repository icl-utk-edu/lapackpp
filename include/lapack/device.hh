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
    int* dev_info, lapack::Queue& queue );

}  // namespace lapack

#endif // LAPACK_DEVICE_HH
