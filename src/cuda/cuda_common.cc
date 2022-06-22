// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "lapack/defines.h"

#if defined(LAPACK_HAVE_CUBLAS)

#include "lapack/device.hh"
#include "cuda_common.hh"

//==============================================================================
namespace blas {

//------------------------------------------------------------------------------
/// @return string describing cuSolver error.
const char* device_error_string( cusolverStatus_t error )
{
    switch (error) {
        case CUSOLVER_STATUS_SUCCESS:
            return "cusolver: success";

        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "cusolver: NOT_INITIALIZED";

        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "cusolver: alloc_failed";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "cusolver: invalid_value";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "cusolver: arch_mismatch";

        case CUSOLVER_STATUS_MAPPING_ERROR:
            return "cusolver: mapping_error";

        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "cusolver: execution_failed";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "cusolver: internal_error";

        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "cusolver: matrix_type_not_supported";

        case CUSOLVER_STATUS_NOT_SUPPORTED:
            return "cusolver: not_supported";

        case CUSOLVER_STATUS_ZERO_PIVOT:
            return "cusolver: zero_pivot";

        case CUSOLVER_STATUS_INVALID_LICENSE:
            return "cusolver: invalid_license";

        case CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED:
            return "cusolver: irs_params_not_initialized";

        case CUSOLVER_STATUS_IRS_PARAMS_INVALID:
            return "cusolver: irs_params_invalid";

        case CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC:
            return "cusolver: irs_params_invalid_prec";

        case CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE:
            return "cusolver: irs_params_invalid_refine";

        case CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER:
            return "cusolver: irs_params_invalid_maxiter";

        case CUSOLVER_STATUS_IRS_INTERNAL_ERROR:
            return "cusolver: irs_internal_error";

        case CUSOLVER_STATUS_IRS_NOT_SUPPORTED:
            return "cusolver: irs_not_supported";

        case CUSOLVER_STATUS_IRS_OUT_OF_RANGE:
            return "cusolver: irs_out_of_range";

        case CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES:
            return "cusolver: irs_nrhs_not_supported_for_refine_gmres";

        case CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED:
            return "cusolver: irs_infos_not_initialized";

        case CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED:
            return "cusolver: irs_infos_not_destroyed";

        case CUSOLVER_STATUS_IRS_MATRIX_SINGULAR:
            return "cusolver: irs_matrix_singular";

        case CUSOLVER_STATUS_INVALID_WORKSPACE:
            return "cusolver: invalid_workspace";

        default:
            return "cusolver: unknown error code";
    }
}

} // namespace blas

#endif // LAPACK_HAVE_CUBLAS
