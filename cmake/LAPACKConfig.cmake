# Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# Check if this file has already been run with these settings.
if (DEFINED lapack_config_cache
    AND "${lapack_config_cache}" STREQUAL "${LAPACK_LIBRARIES}")

    message( DEBUG "LAPACK config already done for '${LAPACK_LIBRARIES}'" )
    return()
endif()
set( lapack_config_cache "${LAPACK_LIBRARIES}" CACHE INTERNAL "" )

include( "cmake/util.cmake" )

#-------------------------------------------------------------------------------
message( STATUS "Checking LAPACK version" )

try_run(
    run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/config/lapack_version.cc"
    LINK_LIBRARIES
        ${LAPACK_LIBRARIES} ${blaspp_libraries}
    COMPILE_DEFINITIONS
        ${blaspp_defines}
    COMPILE_OUTPUT_VARIABLE
        compile_output
    RUN_OUTPUT_VARIABLE
        run_output
)
debug_try_run( "lapack_version.cc" "${compile_result}" "${compile_output}"
                                   "${run_result}" "${run_output}" )

if (compile_result
    AND "${run_output}" MATCHES "LAPACK_VERSION=(([0-9]+)\\.([0-9]+)\\.([0-9]+))")
    # Form version without periods (30201 for 3.2.1) for easy
    # comparisons in C preprocessor.
    set( lapack_version "${CMAKE_MATCH_2}${CMAKE_MATCH_3}${CMAKE_MATCH_4}" )
    message( "${blue}   LAPACK version ${CMAKE_MATCH_1} (${lapack_version})${plain}" )
    list( APPEND lapackpp_defs_ "-DLAPACK_VERSION=${lapack_version}" )
else()
    message( "${red}   Unknown LAPACK version${plain}" )
endif()

#-------------------------------------------------------------------------------
message( STATUS "Checking for XBLAS" )

try_run(
    run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/config/lapack_xblas.cc"
    LINK_LIBRARIES
        ${LAPACK_LIBRARIES} ${blaspp_libraries}
    COMPILE_DEFINITIONS
        ${blaspp_defines}
    COMPILE_OUTPUT_VARIABLE
        compile_output
    RUN_OUTPUT_VARIABLE
        run_output
)
debug_try_run( "lapack_xblas.cc" "${compile_result}" "${compile_output}"
                                 "${run_result}" "${run_output}" )

if (compile_result AND "${run_output}" MATCHES "ok")
    message( "${blue}   Found XBLAS${plain}" )
    list( APPEND lapackpp_defs_ "-DLAPACK_HAVE_XBLAS" )
else()
    message( "${red}   XBLAS not found.${plain}" )
endif()

#-------------------------------------------------------------------------------
# Find LAPACKE, either in the BLAS/LAPACK library or in -llapacke.
# Check for pstrf (Cholesky with pivoting).

set( lib_list ";-llapacke" )
message( DEBUG "lib_list ${lib_list}" )

foreach (lib IN LISTS lib_list)
    message( STATUS "Checking for LAPACKE library ${lib}" )

    try_run(
        run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/config/lapacke_pstrf.cc"
        LINK_LIBRARIES
            ${lib} ${LAPACK_LIBRARIES} ${blaspp_libraries}
        COMPILE_DEFINITIONS
            ${blaspp_defines}
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )
    debug_try_run( "lapacke_pstrf.cc" "${compile_result}" "${compile_output}"
                                      "${run_result}" "${run_output}" )

    if (compile_result AND "${run_output}" MATCHES "ok")
        list( APPEND lapackpp_defs_ "-DLAPACK_HAVE_LAPACKE" )
        set( lapacke_libraries "${lib}" CACHE INTERNAL "" )
        set( lapacke_found true CACHE INTERNAL "" )
        break()
    endif()
endforeach()

if (lapacke_found)
    if (NOT lapacke_libraries)
        message( "${blue}   Found LAPACKE library in BLAS library${plain}" )
    else()
        message( "${blue}   Found LAPACKE library: ${lapacke_libraries}${plain}" )
    endif()
else()
    message( "${red}   LAPACKE library not found. Tester cannot be built.${plain}" )
endif()

#-------------------------------------------------------------------------------
message( DEBUG "
lapackpp_defs_    = '${lapackpp_defs_}'
lapacke_found     = '${lapacke_found}'
lapacke_libraries = '${lapacke_libraries}'
")
