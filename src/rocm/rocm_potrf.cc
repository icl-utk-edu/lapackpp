// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack/defines.h"

#if defined(LAPACK_HAVE_ROCBLAS)

#include "rocm_common.hh"

//==============================================================================
// todo: put into BLAS++ header somewhere.
// changed from blas::device to blas::internal in 5ca8ad35 2022-11-28

namespace blas {
namespace internal {

rocblas_fill uplo2rocblas(blas::Uplo uplo);

} // namespace internal
} // namespace blas

//==============================================================================
namespace lapack {

//------------------------------------------------------------------------------
// Intermediate wrappers around rocSolver to deal with precisions.
void rocsolver_potrf(
    rocblas_handle solver, rocblas_fill uplo, rocblas_int n,
    float* dA, rocblas_int ldda, rocblas_int* info )
{
    rocsolver_spotrf(
        solver, uplo, n, dA, ldda, info );
}

//----------
void rocsolver_potrf(
    rocblas_handle solver, rocblas_fill uplo, rocblas_int n,
    double* dA, rocblas_int ldda, rocblas_int* info )
{
    rocsolver_dpotrf(
        solver, uplo, n, dA, ldda, info );
}

//----------
void rocsolver_potrf(
    rocblas_handle solver, rocblas_fill uplo, rocblas_int n,
    std::complex<float>* dA, rocblas_int ldda, rocblas_int* info )
{
    rocsolver_cpotrf(
        solver, uplo, n,
        (rocblas_float_complex*) dA, ldda, info );
}

//----------
void rocsolver_potrf(
    rocblas_handle solver, rocblas_fill uplo, rocblas_int n,
    std::complex<double>* dA, rocblas_int ldda, rocblas_int* info )
{
    rocsolver_zpotrf(
        solver, uplo, n,
        (rocblas_double_complex*) dA, ldda, info );
}

//------------------------------------------------------------------------------
// Wrapper around rocSolver.
// This is async. Once finished, the return info is in dev_info on the device.
template <typename scalar_t>
void potrf(
    lapack::Uplo uplo, int64_t n,
    scalar_t* dA, int64_t ldda,
    device_info_int* dev_info, lapack::Queue& queue )
{
    // todo: check for overflow
    auto solver = queue.handle();
    auto uplo_ = blas::internal::uplo2rocblas( uplo );

    // for cuda, rocm, call set_device; for oneapi, do nothing.
    blas::internal_set_device( queue.device() );

    rocsolver_potrf( solver, uplo_, n, dA, ldda, dev_info );
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

#endif // LAPACK_HAVE_ROCBLAS
