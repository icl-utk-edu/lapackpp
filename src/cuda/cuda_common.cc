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
            return "cusolver: not initialized";

        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "cusolver: alloc failed";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "cusolver: invalid value";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "cusolver: arch mismatch";

        case CUSOLVER_STATUS_MAPPING_ERROR:
            return "cusolver: mapping error";

        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "cusolver: execution failed";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "cusolver: internal error";

        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "cusolver: matrix type not supported";

        case CUSOLVER_STATUS_NOT_SUPPORTED:
            return "cusolver: not supported";

        case CUSOLVER_STATUS_ZERO_PIVOT:
            return "cusolver: zero pivot";

        case CUSOLVER_STATUS_INVALID_LICENSE:
            return "cusolver: invalid license";

        #if CUSOLVER_VERSION >= 11000
            case CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED:
                return "cusolver: IRS params not initialized";

            case CUSOLVER_STATUS_IRS_PARAMS_INVALID:
                return "cusolver: IRS params invalid";

            case CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC:
                return "cusolver: IRS params invalid precision";

            case CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE:
                return "cusolver: IRS params invalid refine";

            case CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER:
                return "cusolver: IRS params invalid maxiter";

            case CUSOLVER_STATUS_IRS_INTERNAL_ERROR:
                return "cusolver: IRS internal error";

            case CUSOLVER_STATUS_IRS_NOT_SUPPORTED:
                return "cusolver: IRS not supported";

            case CUSOLVER_STATUS_IRS_OUT_OF_RANGE:
                return "cusolver: IRS out of range";

            case CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES:
                return "cusolver: IRS NRHS not supported for refine GMRES";

            case CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED:
                return "cusolver: IRS infos not initialized";

            case CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED:
                return "cusolver: IRS infos not destroyed";

            case CUSOLVER_STATUS_IRS_MATRIX_SINGULAR:
                return "cusolver: IRS matrix singular";

            case CUSOLVER_STATUS_INVALID_WORKSPACE:
                return "cusolver: invalid workspace";
        #endif

        default:
            return "cusolver: unknown error code";
    }
}

} // namespace blas

#endif // LAPACK_HAVE_CUBLAS
