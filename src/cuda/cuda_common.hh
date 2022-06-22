// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_CUDA_COMMON_HH
#define LAPACK_CUDA_COMMON_HH

#include "lapack/device.hh"

//==============================================================================
namespace lapack {

//------------------------------------------------------------------------------
/// CudaTraits<scalar_t>::datatype maps scalar_t to cudaDataType.
template <typename scalar_t>
class CudaTraits;

//----------
// specializations
template<>
class CudaTraits< float > {
public:
    static constexpr cudaDataType datatype = CUDA_R_32F;
};

//----------
template<>
class CudaTraits< double > {
public:
    static constexpr cudaDataType datatype = CUDA_R_64F;
};

//----------
template<>
class CudaTraits< std::complex<float> > {
public:
    static constexpr cudaDataType datatype = CUDA_C_32F;
};

//----------
template<>
class CudaTraits< std::complex<double> > {
public:
    static constexpr cudaDataType datatype = CUDA_C_64F;
};

} // namespace lapack

//==============================================================================
// Inject is_device_error and device_error_string into blas namespace
// for blas_dev_call macros.
// See blaspp/include/blas/device.hh
namespace blas {

inline bool is_device_error( cusolverStatus_t status )
{
    return (status != CUSOLVER_STATUS_SUCCESS);
}

const char* device_error_string( cusolverStatus_t error );

} // namespace blas

#endif // LAPACK_CUDA_COMMON_HH
