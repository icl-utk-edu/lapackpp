// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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

oneapi::mkl::transpose op2onemkl( blas::Op trans );
oneapi::mkl::side side2onemkl( blas::Side side );

} // namespace internal
} // namespace blas

//==============================================================================
namespace lapack {

//------------------------------------------------------------------------------
// Wrapper around workspace query.
// Real uses ormqr, complex uses unmqr.
template <typename scalar_t>
void unmqr_work_size_bytes(
    lapack::Side side, lapack::Op trans,
    int64_t m, int64_t n, int64_t k,
    scalar_t const* dA, int64_t ldda, scalar_t const* dtau,
    scalar_t* dC, int64_t lddc,
    size_t* dev_work_size, size_t* host_work_size,
    lapack::Queue& queue )
{
    // for real, map ConjTrans to Trans; complex doesn't take Trans.
    if (! blas::is_complex_v<scalar_t> && trans == Op::ConjTrans)
        trans = Op::Trans;
    lapack_error_if( blas::is_complex_v<scalar_t> && trans == Op::Trans );

    auto solver = queue.stream();
    auto side_  = blas::internal::side2onemkl( side );
    auto trans_ = blas::internal::op2onemkl( trans );

    // query for workspace size
    int64_t lwork = 0;
    if constexpr (blas::is_complex_v<scalar_t>) {
        blas_dev_call(
            lwork = oneapi::mkl::lapack::unmqr_scratchpad_size<scalar_t>(
                solver, side_, trans_, m, n, k, ldda, lddc ));
    }
    else {
        blas_dev_call(
            lwork = oneapi::mkl::lapack::ormqr_scratchpad_size<scalar_t>(
                solver, side_, trans_, m, n, k, ldda, lddc ));
    }
    *dev_work_size = lwork * sizeof(scalar_t);
    *host_work_size = 0;
}

//------------------------------------------------------------------------------
// Wrapper around oneMKL.
// This is async. Once finished, the return info is in dev_info on the device.
// In oneMKL, there is no info; it is set to 0.
// Real uses ormqr, complex uses unmqr.
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
    // for real, map ConjTrans to Trans; complex doesn't take Trans.
    if (! blas::is_complex_v<scalar_t> && trans == Op::ConjTrans)
        trans = Op::Trans;
    lapack_error_if( blas::is_complex_v<scalar_t> && trans == Op::Trans );

    auto solver = queue.stream();
    auto side_  = blas::internal::side2onemkl( side );
    auto trans_ = blas::internal::op2onemkl( trans );

    // for cuda, rocm, call set_device; for oneapi, do nothing.
    blas::internal_set_device( queue.device() );

    // launch kernel
    int64_t lwork = dev_work_size / sizeof(scalar_t);
    // oneMKL before 2025.0 takes non-const A and tau, which it only reads.
    if constexpr (blas::is_complex_v<scalar_t>) {
        blas_dev_call(
            oneapi::mkl::lapack::unmqr(
                solver, side_, trans_, m, n, k,
                (scalar_t*) dA, ldda, (scalar_t*) dtau, dC, lddc,
                (scalar_t*) dev_work, lwork ));
    }
    else {
        blas_dev_call(
            oneapi::mkl::lapack::ormqr(
                solver, side_, trans_, m, n, k,
                (scalar_t*) dA, ldda, (scalar_t*) dtau, dC, lddc,
                (scalar_t*) dev_work, lwork ));
    }

    blas::device_memset( dev_info, 0, 1, queue );
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

#endif // LAPACK_HAVE_SYCL
